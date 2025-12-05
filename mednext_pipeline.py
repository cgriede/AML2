"""
MedNeXt-based Mitral Valve Segmentation Pipeline (Simplified - 400 lines)

Lean implementation focusing on core ML pipeline:
- Data loading (fixed approach)
- Simple model (U-Net placeholder)
- Hybrid loss (Dice + CrossEntropy)
- Training loop
- Inference & submission
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import gzip
import pandas as pd
from tqdm import tqdm
import os
import sys
from scipy import ndimage

# Add mednext to path
sys.path.insert(0, '/workspaces/AML2/mednext')
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def resize_frame(frame, target_size=(128, 128)):
    """Resize frame using scipy zoom to maintain aspect ratio or fit exactly"""
    h, w = frame.shape[:2]
    if (h, w) == target_size:
        return frame
    
    # Calculate zoom factors
    zoom_h = target_size[0] / h
    zoom_w = target_size[1] / w
    
    # Use nearest neighbor for segmentation masks, linear for images
    order = 0 if frame.ndim == 2 else 0
    resized = ndimage.zoom(frame, (zoom_h, zoom_w) + (1,) * (frame.ndim - 2), order=order)
    return resized


class MitralValveDataset(Dataset):
    """Load temporal sequences from train.pkl - lazy loading, compute on demand"""
    
    def __init__(self, data_list, seq_len=3):
        self.data_list = data_list
        self.seq_len = seq_len
        self.indices = []  # Store (item_idx, frame_idx) pairs
        
        for item_idx, item in enumerate(data_list):
            for center_idx in item['frames']:
                self.indices.append((item_idx, center_idx))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item_idx, center_idx = self.indices[idx]
        item = self.data_list[item_idx]
        
        video = item['video'].astype(np.float32) / 255.0
        label = item['label'].astype(np.float32)
        num_frames = video.shape[2]
        
        # Get seq_len consecutive frames centered on labeled frame
        start_idx = max(0, center_idx - self.seq_len // 2)
        end_idx = min(num_frames, start_idx + self.seq_len)
        start_idx = max(0, end_idx - self.seq_len)
        
        seq = video[:, :, start_idx:end_idx]
        mask_seq = label[:, :, start_idx:end_idx]
        
        if seq.shape[2] < self.seq_len:
            pad_width = ((0, 0), (0, 0), (0, self.seq_len - seq.shape[2]))
            seq = np.pad(seq, pad_width, mode='edge')
            mask_seq = np.pad(mask_seq, pad_width, mode='edge')
        
        # Resize all frames to consistent size (H, W, T)
        seq_resized = np.zeros((128, 128, self.seq_len), dtype=np.float32)
        mask_resized = np.zeros((128, 128, self.seq_len), dtype=np.float32)
        for t in range(self.seq_len):
            seq_resized[:, :, t] = resize_frame(seq[:, :, t], (128, 128))
            mask_resized[:, :, t] = resize_frame(mask_seq[:, :, t], (128, 128))
        
        seq_resized = np.expand_dims(seq_resized, axis=0)
        mask_resized = np.expand_dims(mask_resized, axis=0)
        
        return {
            'frame': torch.from_numpy(seq_resized).float(),
            'mask': torch.from_numpy(mask_resized).float()
        }


class TestDataset(Dataset):
    """Load temporal sequences from test.pkl - lazy loading, compute on demand"""
    
    def __init__(self, data_list, seq_len=3):
        self.data_list = data_list
        self.seq_len = seq_len
        self.indices = []  # Store (item_idx, frame_idx) pairs
        
        for item_idx, item in enumerate(data_list):
            num_frames = item['video'].shape[2]
            for frame_idx in range(num_frames):
                self.indices.append((item_idx, frame_idx))
        
        print(f"[DEBUG][TestDataset] Created dataset with {len(self.indices)} sequences from {len(data_list)} videos")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item_idx, frame_idx = self.indices[idx]
        item = self.data_list[item_idx]
        
        video = item['video'].astype(np.float32) / 255.0
        num_frames = video.shape[2]
        
        # Center sequence on current frame
        start_idx = max(0, frame_idx - self.seq_len // 2)
        end_idx = min(num_frames, start_idx + self.seq_len)
        start_idx = max(0, end_idx - self.seq_len)
        
        seq = video[:, :, start_idx:end_idx]
        
        if seq.shape[2] < self.seq_len:
            pad_width = ((0, 0), (0, 0), (0, self.seq_len - seq.shape[2]))
            seq = np.pad(seq, pad_width, mode='edge')
        
        # Resize all frames to consistent size (H, W, T)
        seq_resized = np.zeros((128, 128, self.seq_len), dtype=np.float32)
        for t in range(self.seq_len):
            seq_resized[:, :, t] = resize_frame(seq[:, :, t], (128, 128))
        
        seq_resized = np.expand_dims(seq_resized, axis=0)
        
        return {
            'frame': torch.from_numpy(seq_resized).float(),
            'name': item['name'],
            'frame_idx': frame_idx
        }


# ============================================================================
# MODEL SETUP
# ============================================================================

class SegmentationNet(nn.Module):
    """MedNeXt Small (2D) - processes frames independently with temporal consistency"""
    
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        # Use MedNeXt Small in 2D mode - process each frame independently
        # Temporal consistency enforced through loss (consecutive predictions should be similar)
        self.model = MedNeXt(
            in_channels=in_channels,
            n_channels=32,              # Small model channels
            n_classes=num_classes,
            exp_r=2,                    # Small model expansion ratio
            kernel_size=3,              # 3x3 kernels (spatial only)
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2,2,2,2,2,2,2,2,2],  # Small model architecture
            dim='2d'                    # 2D mode for frame-by-frame processing
        )
    
    def forward(self, x):
        # x shape: (B, 1, H, W, T) where T is sequence length
        # Process each frame through the model and stack outputs
        b, c, h, w, t = x.shape
        outputs = []
        
        for i in range(t):
            frame = x[:, :, :, :, i]  # (B, 1, H, W)
            out = self.model(frame)   # (B, 2, H, W)
            outputs.append(out)
        
        # Stack outputs: (B, 2, H, W, T)
        output = torch.stack(outputs, dim=-1)
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
        if batch_idx == 0:
            print(f"[DEBUG][train] frames.shape: {frames.shape}, frames.device: {frames.device}")
            print(f"[DEBUG][train] masks.shape: {masks.shape}, masks.device: {masks.device}")
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
            if batch_idx == 0:
                print(f"[DEBUG][val] frames.shape: {frames.shape}, frames.device: {frames.device}")
                print(f"[DEBUG][val] masks.shape: {masks.shape}, masks.device: {masks.device}")
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
            if batch_idx == 0:
                print(f"[DEBUG][predict] frames.shape: {frames.shape}, frames.device: {frames.device}")
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
    
    # Reconstruct full video masks
    results = {}
    for item in test_data:
        name = item['name']
        shape = item['video'].shape
        
        full_mask = np.zeros(shape, dtype=bool)
        for frame_idx, mask in predictions[name].items():
            full_mask[:, :, frame_idx] = mask > 0.5
        
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


def main():
    # TODO: Hyperparameters (add dropout, change LR, batch size, etc.)
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {DEVICE}\n")
    
    # Load data
    train_data = load_zipped_pickle('train.pkl')
    test_data = load_zipped_pickle('test.pkl')
    print(f"Loaded {len(train_data)} training videos and {len(test_data)} test videos\n")

    # Create datasets
    dataset = MitralValveDataset(train_data)
    num_val = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - num_val, num_val])
    test_ds = TestDataset(test_data)
    print(f"Created datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}\n")

    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n")
    
    # Model
    model = SegmentationNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = HybridLoss()
    
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
