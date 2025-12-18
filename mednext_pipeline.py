"""
MedNeXt-based Mitral Valve Segmentation Pipeline (Simplified - 400 lines)

Lean implementation focusing on core ML pipeline:
- Data loading (fixed approach)
- MedNeXt model
- Hybrid loss (Dice + CrossEntropy)
- Training loop
- Inference & submission
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from mednext import create_mednext_v1
from data_prep import load_zipped_pickle, MitralValveDataset
from trainer import SegmentationTrainer
from predictor import SegmentationPredictor


MACHINE_CONFIGS = {
    "codespace_2": {
        "NUM_CPUS": 2,
        "RAM_GB": 8,
        "DEVICE": "cpu",
        "MAX_INPUT_SIZE" : (1, 1, 16, 96, 128),
    },
    "codespace_4": {
        "NUM_CPUS": 4,
        "RAM_GB": 16,
        "DEVICE": "cpu",
        "MAX_INPUT_SIZE" : (2, 1, 16, 64, 80),
        #"MAX_INPUT_SIZE" : (1, 1, 16, 208, 272), debug the double batch
    },
    "surface": {
        "NUM_CPUS": 4,
        "RAM_GB": 8,
        "DEVICE": "cpu",
        "MAX_INPUT_SIZE" : (2, 1, 16, 64, 80),
    },
    "home_station": {
        "NUM_CPUS": 16,
        "RAM_GB": 64,
        "DEVICE": "cuda",
        "MAX_INPUT_SIZE" : (1, 1, 16, 160, 224),  #high performance, for fast dev
        #"MAX_INPUT_SIZE" : (1, 1, 16, 320, 432), #works with high res, 8s / sample
    },
    "hpc_euler_32" : {
        "NUM_CPUS": 32,
        "RAM_GB": 32,
        "DEVICE": "cpu",
        "MAX_INPUT_SIZE" : (2, 1, 16, 208, 272),
    }
}


def setup_machine_config(workspace: str = "home_station"):
    """
    Setup machine-specific configuration and return device, batch size, sequence length, target shape, and num workers.
    This function is called from main to avoid multiprocessing issues with global assignments.
    
    Returns:
        tuple: (DEVICE, batch_size, sequence_length, TARGET_SHAPE, NUM_WORKERS)
    """
    MACHINE_INFO = MACHINE_CONFIGS[workspace]
    
    # MACHINE SPECIFIC SETUP
    if workspace.startswith("hpc"):
        num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    else:
        num_threads = max(1, os.cpu_count() // 2)
    torch.set_num_threads(num_threads)
    NUM_WORKERS = 3 if num_threads > 3 else num_threads
    print(f"PyTorch using {torch.get_num_threads()} threads, {NUM_WORKERS} workers")
    
    # Check CUDA availability before setting device
    requested_device = MACHINE_INFO["DEVICE"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        DEVICE = "cpu"
    else:
        DEVICE = requested_device
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Machine dependent hyperparameters
    batch_size: int = MACHINE_INFO.get("MAX_INPUT_SIZE", (1,1,16,224,304))[0]
    sequence_length: int = MACHINE_INFO.get("MAX_INPUT_SIZE", (1,1,16,224,304))[2]  #default to 16 if not specified
    #(should roughly conserve aspect ratio of original pictures, see data_structures.json) use utils.dimension_scaler to generate sizes if needed
    TARGET_SHAPE: tuple[int, int] = MACHINE_INFO.get("MAX_INPUT_SIZE", (1,1,16,224,304))[3:5]
    
    return DEVICE, batch_size, sequence_length, TARGET_SHAPE, NUM_WORKERS


# ============================================================================
# MODEL SETUP
# ============================================================================
class SegmentationNet(nn.Module):
    """
    3D MedNeXt for MV segmentation.
    - Input: (B, 1, T, H_full, W_full) — full-frame 11-frame clips, normalized [0,1]
    - Output: (B, 1, T, H_full, W_full) — logits for CENTER frame only (sigmoid > 0.5 for mask)
    """
    def __init__(self, in_channels=1, n_frames=16, model_id='S', kernel_size=3, deep_supervision=False):
        super().__init__()
        self.n_frames = n_frames
        
        # Create 3D MedNeXt (spatial_dims=3 treats T as depth)
        self.backbone = create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=1,  # Binary (valve vs bg) — use BCEWithLogitsLoss
            model_id=model_id,  # 'S' small, 'B' base, 'L' large
            kernel_size=kernel_size,  # Depthwise kernel: (3,3,3) start small in time
            deep_supervision=deep_supervision,  # Extra supervision at decoder stages (helps small data)
        )
        
        # We'll handle resizing if needed later (full frames vary in size)

    def forward(self, x):
        # x: (B, C, T, H, W) — e.g., (4, 1, 11, 512, 512) (Batch size, Number of channels, Number of frames, Height, Width)
        #assert x.shape[2] == self.n_frames, f"Expected {self.n_frames} frames, got {x.shape[2]}"
        #assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"
        output = self.backbone(x)

        return output



def main_train():
    # Setup machine configuration (called here to avoid multiprocessing print issues)
    DEVICE, batch_size, sequence_length, TARGET_SHAPE, NUM_WORKERS = setup_machine_config(workspace="home_station")
    
    EVAL = False
    DEBUG = False
    create_video = True

    if DEBUG:
        max_batch_per_epoch = 1
        max_batch_per_val = 1
    else:
        max_batch_per_epoch = 1e3
        max_batch_per_val = 1e3

    
    ######HYPERPARAMETERS######################################################
    learning_rate: float = 1e-3
    n_epochs: int = 20
    model_id: str = 'S'
    rotation_chance: float = 0.0
    random_label_position: bool = False
    print(f"Using target shape: {TARGET_SHAPE}, sequence length {sequence_length}, batch size {batch_size}")
    
    # Load data
    train_data = load_zipped_pickle('train.pkl')
    print(f"Loaded {len(train_data)} training videos\n")
    if EVAL:
        print("Evaluating on test set")
        test_data = load_zipped_pickle('test.pkl')
        print(f"Loaded {len(test_data)} test videos\n")

    # Split the raw data before creating the dataset from sequences
    num_val = int(0.2 * len(train_data))
    # Create generator with correct device to avoid device mismatch error
    train_split, val_split = random_split(train_data, [len(train_data) - num_val, num_val])
    train_ds = MitralValveDataset(train_split, mode="train", target_size=TARGET_SHAPE,
    random_label_position=random_label_position,
    rotation_chance=rotation_chance,
    rotation_angle=20,
    )
    val_ds = MitralValveDataset(val_split, mode="val", target_size=TARGET_SHAPE)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    )
    val_loader = DataLoader(val_ds,
    batch_size=batch_size,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    )
    
    # Quick test instantiation (run this to verify)
    model = SegmentationNet(n_frames=sequence_length, model_id=model_id).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use the new SegmentationTrainer class
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        create_video=create_video,
        max_batch_per_epoch=max_batch_per_epoch,
        max_batch_per_val=max_batch_per_val,
        target_shape=TARGET_SHAPE,
        device=DEVICE
    )
    trainer.train_model()
    return None

def main_predict(model_path: Path, expected_iou: float = 0.5):
    """
    Generate predictions for test set and save submission file.
    
    Args:
        model_path: Path to the trained model checkpoint
        expected_iou: Expected IoU for directory naming (e.g., 0.5430)
    """
    # Setup machine configuration
    DEVICE, batch_size, sequence_length, TARGET_SHAPE, NUM_WORKERS = setup_machine_config(workspace="home_station")
    
    # Load model
    model = SegmentationNet(n_frames=sequence_length, model_id='S').to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load test data
    test_data = load_zipped_pickle('test.pkl')
    #test_data = test_data[:1] #for debug
    print(f"Loaded {len(test_data)} test videos")
    
    # Create test dataset and loader
    # Note: For test data, we need to predict ALL frames, not just labeled ones
    test_ds = MitralValveDataset(test_data, mode="test", seq_len=sequence_length, target_size=TARGET_SHAPE)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=5,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        shuffle=False
    )
    print(f"Test dataset: {len(test_ds)} sequences")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = SegmentationPredictor.predict_test(
        model=model,
        loader=test_loader,
        device=DEVICE,
        seq_len=sequence_length,
        num_postprocess_workers=2,
        test_data=test_data  # Pass test_data to verify original shapes
    )
    
    # Save submission file
    print("Saving submission file...")
    output_file, output_dir = SegmentationPredictor.save_submission(
        predictions=predictions,
        test_data=test_data,
        expected_iou=expected_iou
    )
    generate_videos = False
    if generate_videos:
        # Generate visualization videos
        print("Generating visualization videos...")
        SegmentationPredictor.generate_full_videos(
            predictions=predictions,
            test_data=test_data,
            output_dir=output_dir,
            max_videos=20
        )
        
    print(f"Prediction complete! Submission saved to {output_file}")
    print(f"Videos saved to {output_dir / 'videos'}")
    return predictions, output_file

if __name__ == '__main__':
    main_predict(
        model_path=Path('D:/ETH/Master/AML/AML2/training/20251218_015707_dim160x224/models/ep010_iou05430.pt'),
        expected_iou=0.5430
    )
