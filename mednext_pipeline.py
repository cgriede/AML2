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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm

from mednext import create_mednext_v1
from utils import debug
from data_prep import load_zipped_pickle, resize_frame, MitralValveDataset, TestDataset, DatasetType
from video_generator import VideoGenerator

# ============================================================================
# MODEL SETUP
# ============================================================================
class SegmentationNet(nn.Module):
    """
    3D MedNeXt for MV segmentation.
    - Input: (B, 1, T, H_full, W_full) — full-frame 11-frame clips, normalized [0,1]
    - Output: (B, 1, T, H_full, W_full) — logits for CENTER frame only (sigmoid > 0.5 for mask)
    """
    def __init__(self, in_channels=1, n_frames=11, model_id='S', kernel_size=3, deep_supervision=False):
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

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred_probs, target):
        # Flatten all spatial + temporal dimensions (keep batch and channel)
        pred_flat = pred_probs.view(pred_probs.size(0), pred_probs.size(1), -1)   # → (B, 1, T*H*W)
        target_flat = target.view(target.size(0), target.size(1), -1)  # → (B, 1, T*H*W)
        
        # Compute intersection and union per batch and channel
        intersection = (pred_flat * target_flat).sum(dim=2)  # (B, 1)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # (B, 1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over batch and channel dimensions to get single scalar loss
        return 1 - dice.mean()

class TemporalSmoothLoss(nn.Module):
    """Encourages smooth predictions across consecutive frames using L1 on probabilities"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred):
        """
        pred: (B, C, T, H, W) – probabilities (after sigmoid) or logits
              We recommend applying this on probabilities [0,1] for interpretability.
        """
        # Shift pred by one frame along temporal dimension
        pred_t = pred[:, :, :-1, :, :]   # frames 0 to T-2
        pred_t_plus_1 = pred[:, :, 1:, :, :]  # frames 1 to T-1
        
        # Absolute difference between consecutive frames
        diff = torch.abs(pred_t_plus_1 - pred_t)  # (B, C, T-1, H, W)
        
        # Sum/mean over spatial dims, then over temporal pairs and batch
        if self.reduction == 'mean':
            return diff.mean()
        elif self.reduction == 'sum':
            return diff.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

class AreaConsistencyLoss(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight

    def forward(self, probs):  # probs = sigmoid(logits), shape (B, 1, T, H, W)
        # Sum foreground probability mass per frame in batch
        areas = probs.sum(dim=[2,3,4])  # → (B,)
        # Penalize large changes between consecutive frames in batch (approximation)
        if areas.shape[0] > 1:
            diff = torch.abs(areas[1:] - areas[:-1])
            return self.weight * diff.mean()
        return 0.0 * areas.sum()  # 0 if batch=1

@torch.compile
class CombinedLoss(nn.Module):
    def __init__(self,
      dice_weight: float=50,
      bce_weight : float=50,
      tsl_weight : float=1,
      ac_weight  : float=0.1
    ) -> None    : 
        super().__init__()
        self.dice = BinaryDiceLoss()  # Keep your existing one (it works on probs)
        self.bce = nn.BCEWithLogitsLoss()  # Expects raw logits!
        self.tsl = TemporalSmoothLoss()
        self.ac = AreaConsistencyLoss()
        total_weight = dice_weight + bce_weight + tsl_weight + ac_weight #automatically normalize the weights to 1
        self.dice_weight = dice_weight / total_weight
        self.bce_weight = bce_weight / total_weight
        self.tsl_weight = tsl_weight / total_weight
        self.ac_weight = ac_weight / total_weight

    def forward(self, logits, target, label_idx):
        pred_label = slice_tensor_at_label(logits, label_idx)
        mask_label = slice_tensor_at_label(target, label_idx)
        probs_label = torch.sigmoid(pred_label)  # For Dice
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs_label, mask_label)
        bce_loss = self.bce(pred_label, mask_label.float())
        temp_loss = self.tsl(probs)
        ac_loss = self.ac(probs)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss + self.tsl_weight * temp_loss + self.ac_weight * ac_loss


def compute_iou(pred_probs, target):
    """Jaccard Index (IoU)"""
    pred_binary = (pred_probs > 0.5).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = (pred_binary + target_binary).sum() - intersection
    
    return (intersection + 1e-7) / (union + 1e-7)


def slice_tensor_at_label(pred: torch.Tensor, label_idx: list[int]) -> torch.Tensor:
    slices = []
    for i in range(pred.shape[0]):
        indices = label_idx[i]
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, device=pred.device, dtype=torch.long)
        # pred[i:i+1] keeps batch dim for cat compatibility
        slices.append(pred[i:i+1, :, indices, :, :])   # (1, C, N_i, H, W)

    return torch.cat(slices, dim=0)  # (N_total, C, H, W)

@torch.compile
def train_epoch(model, loader, optimizer, loss_fn, max_batch_per_epoch: int = 1e3):
    model.train()
    total_loss = 0.0
    batches_per_epoch = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc='Train')):
        batches_per_epoch += 1
        if batches_per_epoch > max_batch_per_epoch:
            break

        frames = batch['frame']
        target = batch['mask']
        label_idx = batch['label_idx']
        optimizer.zero_grad()
        try:
            pred = model(frames)
        except Exception as e:
            print(f"[ERROR][train] model forward failed: {e}")
            raise
        #get the labeled slice only for simple evalutaion of the dice loss
        #NOTE: might adjust this if we take the temporal loss into account

        loss = loss_fn(pred, target, label_idx)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    
    return total_loss / len(loader)

@torch.compile
def train_model(model, train_loader, val_loader, optimizer, loss_fn, n_epochs: int,
                create_video: bool=True,
                max_batch_per_epoch: int = 1e3,
                max_batch_per_val:int = 1e3,
                ):
    best_iou = 0.0
    patience = 0
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, max_batch_per_epoch)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}")
        val_loss, val_iou, video_payload = validate(model, val_loader, loss_fn, create_video, max_batch_per_val)
        print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
            if create_video and video_payload is not None:
                try:
                    vgen = VideoGenerator(
                        video_payload["batch"],
                        video_payload["pred_masks"],
                        video_payload["true_masks"],
                        batch_idx=video_payload["batch_idx"],
                        output_dir="videos",
                    )
                    vgen.save_sequence_gif(fps=10, alpha=0.5, frame_skip=1)
                except Exception as e:
                    print(f"[ERROR][train] video generation failed:\n {e}")
        else:
            patience += 1
            if patience >= 50: #TODO: make this a hyperparameter
                print("Early stopping")
                break
    return model


@torch.compile
def validate(model, loader, loss_fn, create_video: bool=True, max_batch_per_val:int = 1e3):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    batches_per_val = 0
    video_payload = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Val')):
            batches_per_val += 1
            if batches_per_val > max_batch_per_val:
                break
            frames = batch['frame']
            target = batch['mask']
            label_idx = batch['label_idx']
            try:
                pred = model(frames)  # (B, C, T, H, W)
            except Exception as e:
                print(f"[ERROR][val] model forward failed: {e}")
                raise

            loss = loss_fn(pred, target, label_idx)
            total_loss += loss.item()

            pred_probs = torch.sigmoid(slice_tensor_at_label(pred, label_idx))
            
            iou = compute_iou(pred_probs, slice_tensor_at_label(target, label_idx))
            total_iou += iou.item()
            if create_video and video_payload is None:
                # Capture first available batch for potential video generation
                video_payload = {
                    "batch": batch,
                    "pred_masks": pred.cpu().numpy(),
                    "true_masks": target.cpu().numpy(),
                    "batch_idx": batch_idx,
                }

    return total_loss / len(loader), total_iou / len(loader), video_payload


def predict_test(model, loader, test_data):
    raise NotImplementedError("Predict test is not implemented by human")
    """Generate predictions for test set - extract center frame from 3D output"""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Predict')):
            frames = batch['frame']
            names = batch['name']
            frame_indices = batch['frame_idx'].cpu().numpy()
            try:
                pred = model(frames)  # Shape: (B, C, T, H, W)
            except Exception as e:
                print(f"[ERROR][predict] model forward failed: {e}")
                raise
            pred_label = slice_tensor_at_label(pred, label_idx)
            
            for i, name in enumerate(names):
                if name not in predictions:
                    predictions[name] = {}
                
                predictions[name][int(frame_indices[i])] = pred_label[i]
    
    # Reconstruct full video masks at ORIGINAL resolution
    results = {}
    for item in test_data:
        name = item['name']
        orig_shape = item['video'].shape  # Original video shape (H, W, T)
        
        full_mask = np.zeros(orig_shape, dtype=bool)
        for frame_idx, mask_112 in predictions[name].items():
            # Resize from 112x112 back to original size
            mask_orig = resize_frame(mask_112, (orig_shape[0], orig_shape[1]))
            full_mask[:, :, frame_idx] = mask_orig > 0.5
        
        results[name] = full_mask
    
    return results


def save_submission(predictions, test_data, output_file='submission.csv'):
    raise NotImplementedError("Save submission is not implemented by human")
    """Save predictions as CSV submission"""
    rows = []
    
    for i, item in enumerate(test_data):
        name = item['name']
        mask = predictions[name]
        
        flattened = mask.flatten()
        indices = np.where(flattened)[0]
        
        rle_parts = []
        if len(indices) > 0:
            start = indices[0]
            length = 1
            for j in range(1, len(indices)):
                if indices[j] == indices[j-1] + 1:
                    length += 1
                else:
                    rle_parts.append([int(start), int(length)])
                    start = indices[j]
                    length = 1
            rle_parts.append([int(start), int(length)])
        
        rows.append({'id': f"{name}_{i}", 'value': str(rle_parts)})
    
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


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
    },
    "hpc_euler_32" : {
        "NUM_CPUS": 32,
        "RAM_GB": 32,
        "DEVICE": "cpu",
        "MAX_INPUT_SIZE" : (2, 1, 16, 208, 272),
    }
}

WORKSPACE = "hpc_euler_32"
MACHINE_INFO = MACHINE_CONFIGS[WORKSPACE]

EVAL = False
DEBUG = False
GB_RAM_PER_BATCH = 16
create_video = True

def main():
    #MACHINE SPECIFIC SETUP
    num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(num_threads)
    print(f"PyTorch using {torch.get_num_threads()} threads")
    torch.set_default_device(MACHINE_INFO["DEVICE"])
    print(f"Device: {MACHINE_INFO['DEVICE']}")

    if DEBUG:
        max_batch_per_epoch = 1
        max_batch_per_val = 1
    else:
        max_batch_per_epoch = 1e3
        max_batch_per_val = 1e3

    
    # TODO: Hyperparameters (add dropout, change LR, batch size, etc.)
    learning_rate: float = 1e-3
    n_epochs: int = 100
    model_id: str = 'S'

    #machine dependant hyperparameters
    batch_size: int = MACHINE_INFO.get("MAX_INPUT_SIZE", (1,1,16,224,304))[0]
    sequence_length: int = MACHINE_INFO.get("MAX_INPUT_SIZE", (1,1,16,224,304))[2]  #default to 16 if not specified
    #(should roughly conserve aspect ratio of original pictures, see data_structures.json) use utils.dimension_scaler to generate sizes if needed
    TARGET_SHAPE: tuple[int, int] = MACHINE_INFO.get("MAX_INPUT_SIZE", (1,1,16,224,304))[3:5]

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
    train_split, val_split = random_split(train_data, [len(train_data) - num_val, num_val])
    train_ds = MitralValveDataset(train_split, target_size=TARGET_SHAPE)
    val_ds = MitralValveDataset(val_split, target_size=TARGET_SHAPE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    if EVAL:
        test_ds = TestDataset(test_data, target_size=TARGET_SHAPE)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        print(f"Test: {len(test_ds)}")
    
    # Quick test instantiation (run this to verify)
    model = SegmentationNet(n_frames=sequence_length, model_id=model_id).compile()
    loss_fn = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = train_model(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, create_video, max_batch_per_epoch, max_batch_per_val)
    return None


if __name__ == '__main__':
    main()
