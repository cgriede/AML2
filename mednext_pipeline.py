"""
MedNeXt-based Mitral Valve Segmentation Pipeline (Simplified - 400 lines)

Lean implementation focusing on core ML pipeline:
- Data loading (fixed approach)
- MedNeXt model
- Hybrid loss (Dice + CrossEntropy)
- Training loop
- Inference & submission
"""

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


# ============================================================================
# MODEL SETUP
# ============================================================================

class SegmentationNet(nn.Module):
    """MedNeXt Small (2D) - processes frames independently with temporal consistency"""
    
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        # For binary segmentation (MV vs background)
        model = create_mednext_v1(
            num_input_channels=1,      # Grayscale ultrasound frames
            num_classes=1,             # Binary: sigmoid output
            model_id='S',
            kernel_size=3,             # Start small (3x3x3 depthwise)
            deep_supervision=True      # Helps training on small data
        )
        model = model.cuda() if torch.cuda.is_available() else model
        print(model)  # Inspect layers
    
    def forward(self, x):
        #define input output behaviour
        return output


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_prob = torch.softmax(pred, dim=1)
        pred_flat = pred_prob.view(pred_prob.size(0), pred_prob.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TemporalSmoothLoss(nn.Module):
    """Encourages smooth predictions across consecutive frames"""
    def forward(self, pred):
        # pred shape: (B, 2, H, W, T)
        if pred.shape[-1] < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Compare consecutive frame predictions
        smooth_loss = 0.0
        for t in range(pred.shape[-1] - 1):
            diff = (pred[:, :, :, :, t] - pred[:, :, :, :, t+1]).abs().mean()
            smooth_loss += diff
        
        return smooth_loss / max(1, pred.shape[-1] - 1)


class HybridLoss(nn.Module):
    def __init__(self, temporal_weight=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.temporal_loss = TemporalSmoothLoss()
        self.temporal_weight = temporal_weight
    
    def forward(self, pred, target):
        # pred shape: (B, 2, H, W, T), target shape: (B, 1, H, W, T)
        
        # For CE/Dice, we average over temporal dimension
        pred_flat = pred.view(-1, pred.shape[1], pred.shape[2], pred.shape[3])  # (B*T, 2, H, W)
        target_flat = target.view(-1, 1, target.shape[2], target.shape[3])      # (B*T, 1, H, W)
        
        target_ce = target_flat.squeeze(1).long()
        ce = self.ce_loss(pred_flat, target_ce)
        dice = self.dice_loss(pred_flat, target_flat)
        
        # Add temporal smoothness constraint
        temporal = self.temporal_loss(pred)
        
        return 0.45 * ce + 0.45 * dice + self.temporal_weight * temporal


def compute_iou(pred, target):
    """Jaccard Index (IoU)"""
    pred_binary = (pred > 0.5).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = (pred_binary + target_binary).sum() - intersection
    
    return (intersection + 1e-7) / (union + 1e-7)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc='Train')):
        frames = batch['frame'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        try:
            pred = model(frames)
        except Exception as e:
            print(f"[ERROR][train] model forward failed: {e}")
            raise
        loss = loss_fn(pred, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Val')):
            frames = batch['frame'].to(device)
            masks = batch['mask'].to(device)
            try:
                pred = model(frames)  # (B, 2, H, W, T)
            except Exception as e:
                print(f"[ERROR][val] model forward failed: {e}")
                raise
            loss = loss_fn(pred, masks)
            total_loss += loss.item()
            
            # Use center frame for IoU evaluation
            center_idx = pred.shape[-1] // 2
            pred_center = pred[:, :, :, :, center_idx]  # (B, 2, H, W)
            masks_center = masks[:, :, :, :, center_idx]  # (B, 1, H, W)
            
            pred_probs = torch.softmax(pred_center, dim=1)
            pred_valve = pred_probs[:, 1]
            
            iou = compute_iou(pred_valve, masks_center.squeeze(1))
            total_iou += iou.item()
    
    return total_loss / len(loader), total_iou / len(loader)


def predict_test(model, loader, device, test_data):
    """Generate predictions for test set - extract center frame from 3D output"""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Predict')):
            frames = batch['frame'].to(device)
            names = batch['name']
            frame_indices = batch['frame_idx'].cpu().numpy()
            try:
                pred = model(frames)  # Shape: (B, 2, H, W, T)
            except Exception as e:
                print(f"[ERROR][predict] model forward failed: {e}")
                raise
            # Extract center frame from temporal sequence
            center_idx = pred.shape[-1] // 2
            pred_center = pred[:, :, :, :, center_idx]  # (B, 2, H, W)
            
            pred_probs = torch.softmax(pred_center, dim=1)
            pred_valve = pred_probs[:, 1].cpu().numpy()
            
            for i, name in enumerate(names):
                if name not in predictions:
                    predictions[name] = {}
                
                predictions[name][int(frame_indices[i])] = pred_valve[i]
    
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


MACHINE_INFO = {}
WORKSPACE = "codespace-2"


if WORKSPACE == "codespace-2":
    MACHINE_INFO = {
        "NUM_CPUS" : 2,
        "RAM_GB" : 8,
        "DEVICE" : "cpu"
    }
if WORKSPACE == "home_station":
    MACHINE_INFO = {
        "NUM_CPUS" : 16,
        "RAM_GB" : 64,
        "DEVICE" : "cuda"
    }


def main():
    DEVICE = MACHINE_INFO["DEVICE"]
    print(f"Device: {DEVICE}\n")
    
    # TODO: Hyperparameters (add dropout, change LR, batch size, etc.)
    learning_rate = 1e-4
    n_epochs = 1

    #machine dependant hyperparameters
    batch_size = 4
    
    # Load data
    train_data = load_zipped_pickle('train.pkl')
    test_data = load_zipped_pickle('test.pkl')
    print(f"Loaded {len(train_data)} training videos and {len(test_data)} test videos\n")

    for key in test_data[0]:
        print(f"Key: {key}, Type: {type(test_data[0][key])}")
    
    # Print video sizes for debugging
    print("Training video sizes:")
    train_shapes = {}
    for i, item in enumerate(train_data):
        shape = item['video'].shape
        shape = shape[:2]
        if shape not in train_shapes:
            train_shapes[shape] = 0
        train_shapes[shape] += 1
    for key, value in train_shapes.items():
        print(f"  Shape {key}: {value} videos")
    print()
    
    print("Test video sizes:")
    test_shapes = {}
    for i, item in enumerate(test_data):
        shape = item['video'].shape
        shape = shape[:2]
        if shape not in test_shapes:
            test_shapes[shape] = 0
        test_shapes[shape] += 1
    for key, value in test_shapes.items():
        print(f"  Shape {key}: {value} videos")
    
    print()
    #TODO: choose target size for reshape (make adjustable)
    TARGET_SHAPE = (384, 512)
    #TODO: choose input temporal design (make adjustable)

    #TODO: Create datasets
    # Training data: use original 112x112 size (no resize needed)
    dataset = MitralValveDataset(train_data, target_size=TARGET_SHAPE)
    num_val = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - num_val, num_val])

    #TODO: define the model

    #TODO: evaluate the

    
    # Test data: resize to 112x112 to match training
    test_ds = TestDataset(test_data, target_size=(112, 112))
    print(f"Created datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n")
    
    #TODO: model init, instantiate a model
    # Model
    model = SegmentationNet().to(DEVICE)
    #TODO: define the optimizer
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    #TODO: define the LossFn
    #loss_fn = HybridLoss()
    
    # Training loop
    best_iou = 0.0
    patience = 0
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_iou = validate(model, val_loader, loss_fn, DEVICE)
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, IoU={val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience += 1
            if patience >= 20:
                print("Early stopping")
                break
    
    print(f"\nBest IoU: {best_iou:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Predict
    predictions = predict_test(model, test_loader, DEVICE, test_data)
    
    # Submit
    save_submission(predictions, test_data)


if __name__ == '__main__':
    main()
