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
from utils import write_configuration_string

from mednext import create_mednext_v1, upkern_load_weights
from data_prep import load_zipped_pickle, MitralValveDataset, DatasetType, DynamicAmateurMixDataset
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
        "MAX_INPUT_SIZE" : (1, 1, 32, 240, 320),
        #"MAX_INPUT_SIZE" : (1, 1, 16, 320, 432), #works with high res, 8s / sample
        #"MAX_INPUT_SIZE" : (1, 1, 16, 64, 80),  #high performance, for fast dev
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
    NUM_WORKERS = 4 if num_threads >= 4 else num_threads
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
        shapes =  x.shape[-3 : -1]
        for shape in shapes:
            if shape % 16 != 0:
                raise ValueError(f"Expected shape to be divisible by 16, got {shape}")
        output = self.backbone(x)

        return output


def main_train_expert():

    # Setup machine configuration (called here to avoid multiprocessing print issues)
    DEVICE, batch_size, sequence_length, TARGET_SHAPE, NUM_WORKERS = setup_machine_config(workspace="home_station")
    
    DEBUG = False
    create_video = True

    if DEBUG:
        max_batch_per_epoch = 1
        max_batch_per_val = 1
        NUM_WORKERS = 1
    else:
        max_batch_per_epoch = 1e3
        max_batch_per_val = 1e3

    ######HYPERPARAMETERS######################################################
    HYPERPARAMETERS = {
        "upkernel_lr": (5e-4, 2e-4, 7e-5),
        "n_epochs_expert": (40, 40, 50),
        "model_id": 'S',
        "random_label_position": True,
        "rotation_chance": 0.2,
        "upkernel_size": (3, 5, 7),
        "deep_supervision": True,
        "sequence_length": sequence_length,
    }
    print(f"Using target shape: {TARGET_SHAPE}, sequence length {sequence_length}, batch size {batch_size}")
    n_epochs_expert = HYPERPARAMETERS["n_epochs_expert"]
    model_id = HYPERPARAMETERS["model_id"]
    rotation_chance = HYPERPARAMETERS["rotation_chance"]
    random_label_position = HYPERPARAMETERS["random_label_position"]
    deep_supervision = HYPERPARAMETERS["deep_supervision"]
    upkernel_lr = HYPERPARAMETERS["upkernel_lr"]
    sequence_length = HYPERPARAMETERS["sequence_length"]
    ############################################################
    #get the pretrained model from the amateur training
    # Load data
    train_data = load_zipped_pickle('train.pkl')
    print(f"Loaded {len(train_data)} training videos\n")
    train_data_expert = [item for item in train_data if item.get("dataset") == "expert"]
    train_split_ex, val_split_ex = random_split(train_data_expert, [0.9, 0.1])

    train_ds_expert = MitralValveDataset(train_split_ex, mode="train", target_size=TARGET_SHAPE,
    random_label_position=random_label_position,
    rotation_chance=rotation_chance,
    rotation_angle=20,
    dataset_type=DatasetType.EXPERT,
    seq_len=sequence_length,
    )
    val_ds_expert = MitralValveDataset(val_split_ex, mode="val", target_size=TARGET_SHAPE)

    train_loader_expert = DataLoader(train_ds_expert,
    batch_size=batch_size,
    shuffle=True,
    num_workers=6,
    persistent_workers=True,
    prefetch_factor=5,
    )
    val_loader_expert = DataLoader(val_ds_expert,
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    )

    train_data_amateur = [item for item in train_data if item.get("dataset") == "amateur"]
    train_ds_amateur = MitralValveDataset(train_data_amateur, mode="train", target_size=None, #no rescaling since 112x112 shape
    random_label_position=random_label_position,
    rotation_chance=rotation_chance,
    rotation_angle=20,
    dataset_type=DatasetType.AMATEUR,
    seq_len=sequence_length,
    )

    # Create dynamic dataset
    dynamic_dataset = DynamicAmateurMixDataset(
        expert_dataset=train_ds_expert,
        amateur_dataset=train_ds_amateur,
        num_amateur_per_epoch=5,   # your target
        seed=42  # for reproducibility
    )

    mixed_loader = DataLoader(
        dynamic_dataset,
        batch_size=batch_size,
        shuffle=True,                  # shuffles within current subset
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
    )

    model_3 = SegmentationNet(n_frames=sequence_length, model_id=model_id, kernel_size=3, deep_supervision=deep_supervision).to(DEVICE)
    model_5 = SegmentationNet(n_frames=sequence_length, model_id=model_id, kernel_size=5, deep_supervision=deep_supervision).to(DEVICE)
    model_7 = SegmentationNet(n_frames=sequence_length, model_id=model_id, kernel_size=7, deep_supervision=deep_supervision).to(DEVICE)
    optimizer_3 = optim.Adam(model_3.parameters(), lr=upkernel_lr[0])
    optimizer_5 = optim.Adam(model_5.parameters(), lr=upkernel_lr[1])
    optimizer_7 = optim.Adam(model_7.parameters(), lr=upkernel_lr[2])

    models = [model_3, model_5, model_7]
    optimizers = [optimizer_3, optimizer_5, optimizer_7]
    train_loaders = [mixed_loader, train_loader_expert, train_loader_expert]
    for i, (model, optimizer, train_loader, n_epochs) in enumerate(zip(models, optimizers, train_loaders, n_epochs_expert)):
        # Use the new SegmentationTrainer expert class
        trainer_expert = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader_expert,
            optimizer=optimizer,
            n_epochs=n_epochs,
            create_video=create_video,
            max_batch_per_epoch=max_batch_per_epoch,
            max_batch_per_val=max_batch_per_val,
            target_shape=TARGET_SHAPE,
            device=DEVICE,
            deep_supervision=deep_supervision,
            save_threshold_iou=0.4,
            description=(f"EXPERT TRAINING KERNEL 3-5-7 upsizing current run {i}\n" + write_configuration_string(HYPERPARAMETERS))
        )
        trainer_expert.train_model()
        if i < len(models)-1:
            #do the upkernel trick on the best model
            model.load_state_dict(torch.load(trainer_expert.best_model_path_mixed, map_location=DEVICE))
            upkern_load_weights(models[i+1], model)
        else:
            main_predict(trainer_expert.best_model_path_mixed, expected_iou=0.5, probability_threshold=0.5, path_notes=f"full cycle prediction")
            best_model_path_finetune = main_train_finetune_no_val(trainer_expert.best_model_path_mixed)
            main_predict(best_model_path_finetune, expected_iou=0.5, probability_threshold=0.5, path_notes=f"full cycle prediction with finetune")

def main_train_finetune_no_val(model_path: Path):
        # Setup machine configuration (called here to avoid multiprocessing print issues)
    DEVICE, batch_size, sequence_length, TARGET_SHAPE, NUM_WORKERS = setup_machine_config(workspace="home_station")
    
    DEBUG = False
    create_video = False

    if DEBUG:
        max_batch_per_epoch = 1
    else:
        max_batch_per_epoch = 1e3

    ######HYPERPARAMETERS######################################################
    HYPERPARAMETERS = {
        "upkernel_lr": 5e-5,
        "n_epochs_expert": 20,
        "model_id": 'S',
        "random_label_position": True,
        "rotation_chance": 0.0,
        "upkernel_size": 7,
        "deep_supervision": True,
    }
    n_epochs_expert = HYPERPARAMETERS["n_epochs_expert"]
    model_id = HYPERPARAMETERS["model_id"]
    rotation_chance = HYPERPARAMETERS["rotation_chance"]
    random_label_position = HYPERPARAMETERS["random_label_position"]
    upkernel_size = HYPERPARAMETERS["upkernel_size"]
    deep_supervision = HYPERPARAMETERS["deep_supervision"]
    upkernel_lr = HYPERPARAMETERS["upkernel_lr"]
    ############################################################
    #get the pretrained model from the amateur training

    model = SegmentationNet(model_id=model_id, kernel_size=upkernel_size, deep_supervision=deep_supervision).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=upkernel_lr)
    # Load data
    train_data = load_zipped_pickle('train.pkl')
    print(f"Loaded {len(train_data)} training videos\n")
    train_data_expert = [item for item in train_data if item.get("dataset") == "expert"]

    train_ds_expert = MitralValveDataset(train_data_expert, mode="train", target_size=TARGET_SHAPE,
    random_label_position=random_label_position,
    rotation_chance=rotation_chance,
    rotation_angle=20,
    dataset_type=DatasetType.EXPERT
    )

    train_loader_expert = DataLoader(train_ds_expert,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    )

    # Use the new SegmentationTrainer expert class
    trainer_expert = SegmentationTrainer(
        model=model,
        train_loader=train_loader_expert,
        val_loader=None,
        optimizer=optimizer,
        n_epochs=n_epochs_expert,
        create_video=create_video,
        max_batch_per_epoch=max_batch_per_epoch,
        max_batch_per_val=0,
        target_shape=TARGET_SHAPE,
        device=DEVICE,
        deep_supervision=deep_supervision,
        save_threshold_iou=0.5,
        no_validation=True,
        description=("EXPERT TRAINING Full Data Finetune\n" + write_configuration_string(HYPERPARAMETERS))
    )
    trainer_expert.train_model()
    return trainer_expert.best_model_path_train

def main_predict(model_path: Path, expected_iou: float = 0.5, probability_threshold: float = 0.5, path_notes: str = ""):
    """
    Generate predictions for test set and save submission file.
    
    Args:
        model_path: Path to the trained model checkpoint
        expected_iou: Expected IoU for directory naming (e.g., 0.5430)
    """
    # Setup machine configuration
    DEVICE, batch_size, sequence_length, TARGET_SHAPE, NUM_WORKERS = setup_machine_config(workspace="home_station")
    
    # Load model
    model = SegmentationNet(n_frames=sequence_length, model_id='S', kernel_size=7, deep_supervision=True).to(DEVICE)
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
    predictor = SegmentationPredictor(
        model=model,
        loader=test_loader,
        device=DEVICE,
        seq_len=sequence_length,
        num_postprocess_workers=2,
        test_data=test_data,
        deep_supervision=True,
        probability_threshold=probability_threshold
    )
    predictions = predictor.predict_test()
    
    # Save submission file
    print("Saving submission file...")
    output_file, output_dir = SegmentationPredictor.save_submission(
        predictions=predictions,
        test_data=test_data,
        expected_iou=expected_iou,
        path_notes=path_notes
    )

    cmd = f'kaggle competitions submit -c eth-aml-2025-project-task-2 -f {output_file} -m "{path_notes + " " + str(expected_iou)}"'
    os.system(cmd)
    print(f"Submission sent to Kaggle")
    generate_videos = True
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


def release_to_kaggle(csv_file: Path, message: str = ""):
    import sys
    from pathlib import Path
    from kaggle.api.kaggle_api_extended import KaggleApi
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        return
    
    api = KaggleApi()
    api.print_config_values()
    api.set_config_value('username', 'cedricgrieder')
    api.set_config_value('key', 'KGAT_aeac930ab61b3d68d83c8c57b24dbb17')
    os.environ['KAGGLE_USERNAME'] = 'cedricgrieder'   # Replace with your actual username
    os.environ['KAGGLE_KEY'] = 'KGAT_aeac930ab61b3d68d83c8c57b24dbb17'
    try:
        api.authenticate()  # Will raise if credentials are missing/invalid
    except Exception as e:
        print("Authentication failed:")
        print(e)
        return
    
    try:
        result = api.competition_submit_cli(
            file_name=str(csv_file),
            message=message or "Submitted via Python API",
            competition="eth-aml-2025-project-task-2",
            quiet=False  # Set to True for less output
        )
        print("Submission successful!")
        print(result)  # Prints submission details
    except Exception as e:
        print("Submission failed:")
        print(e)


if __name__ == '__main__':
    mode = "train_expert"
    if mode == "train_expert":
        main_train_expert()
    elif mode == "train_finetune_no_val":
        main_train_finetune_no_val()
    elif mode == "predict":
        thresholds = [0.5]
        for threshold in thresholds:
            print(f"Predicting with probability threshold {threshold}")
            main_predict(
                model_path=Path(r'D:\ETH\Master\AML\AML2\training\20251218_193011_dim160x224\models\ep086_iouiou06163.pt'),
                expected_iou=0.6163,
                probability_threshold=threshold,
                path_notes=f"threshold_045"
            )
    elif mode == "release_to_kaggle":
        release_to_kaggle(csv_file=Path(r'D:\ETH\Master\AML\AML2\results\20251218_05430\submission.csv'), message="automatic release test")
