from utils import slice_tensor_at_label
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
import torch.nn as nn
from video_generator import VideoGenerator

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
    def __init__(self, weight=0.01):
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


class SegmentationTrainer:

    allow_no_user_input: bool = False

    class CombinedLoss(nn.Module):
        def __init__(self,
        dice_weight: float=50,
        bce_weight : float=50,
        tsl_weight : float=1,
        ac_weight  : float=0.5
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
            # Access parent class static method
            pred_label = slice_tensor_at_label(logits, label_idx)
            mask_label = slice_tensor_at_label(target, label_idx)
            probs_label = torch.sigmoid(pred_label)  # For Dice
            probs = torch.sigmoid(logits)
            dice_loss = self.dice(probs_label, mask_label)
            bce_loss = self.bce(pred_label, mask_label.float())
            temp_loss = self.tsl(probs)
            ac_loss = self.ac(probs)
            return self.dice_weight * dice_loss + self.bce_weight * bce_loss + self.tsl_weight * temp_loss + self.ac_weight * ac_loss

    @staticmethod
    def compute_iou(pred_probs, target):
        """Jaccard Index (IoU)"""
        pred_binary = (pred_probs > 0.5).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = (pred_binary + target_binary).sum() - intersection
        
        return (intersection + 1e-7) / (union + 1e-7)


    def __init__(self, model, train_loader, val_loader, optimizer, n_epochs: int,
                loss_fn: nn.Module = CombinedLoss(),
                create_video: bool=True,
                max_batch_per_epoch: int = 1e3,
                max_batch_per_val:int = 1e3,
                patience: int = 50,
                save_threshold_iou: float = 0.4,
                training_dir: Path = None,
                target_shape: tuple = None,
                device: str = "cpu",
                ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.create_video = create_video
        self.max_batch_per_epoch = max_batch_per_epoch
        self.max_batch_per_val = max_batch_per_val
        self.patience = patience
        self.save_threshold_iou = save_threshold_iou
        self.device = device
        # Initialize training summary DataFrame
        self.training_summary = []
        
        # Set up training directory
        if training_dir is None:
            if target_shape is None:
                raise ValueError("target_shape must be provided if training_dir is None")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.training_dir = Path('training') / f"{timestamp}_dim{target_shape[0]}x{target_shape[1]}"
        else:
            self.training_dir = Path(training_dir)
        
        # Create training directory structure
        self.training_dir.mkdir(parents=True, exist_ok=True)
        (self.training_dir / 'models').mkdir(exist_ok=True)
        (self.training_dir / 'videos').mkdir(exist_ok=True)
        
        if not self.allow_no_user_input:
            self._get_user_training_description()
        else:
            print("Allowing no user input, continuing without user input")
        
    def _get_user_training_description(self):
        description = input("Enter a description for the training: ")
        with open(self.training_dir / 'description.txt', 'w') as f:
            f.write(description)
        print(f"Training description saved to {self.training_dir / 'description.txt'}")
        return description

    @staticmethod
    def _train_epoch(model, loader, optimizer, loss_fn, device, max_batch_per_epoch: int = 1e3):
        model.train()
        total_loss = 0.0
        batches_per_epoch = 0
        
        for batch_idx, batch in enumerate(loader):
            batches_per_epoch += 1
            if batches_per_epoch > max_batch_per_epoch:
                break

            frames = batch['frame'].to(device)
            target = batch['mask'].to(device)
            label_idx = batch['label_idx'].to(device)
            optimizer.zero_grad()
            try:
                pred = model(frames)
            except Exception as e:
                print(f"[ERROR][train] model forward failed: {e}")
                raise

            loss = loss_fn(pred, target, label_idx)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        
        return total_loss / len(loader)

    def train_model(self):
        best_iou = 0.0
        patience = 0
        best_model_path = None
        
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(self.model, self.train_loader, self.optimizer, self.loss_fn, self.device, self.max_batch_per_epoch)
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}")
            val_loss, val_iou, val_iou_orig, video_payload = self._validate(self.model, self.val_loader, self.loss_fn, self.device, self.create_video, self.max_batch_per_val)
            print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}, Val IoU (orig size)={val_iou_orig:.4f}")
            
            # Record training summary
            self.training_summary.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou
            })
            
            if val_iou > best_iou:
                best_iou = val_iou
                patience = 0
                
                # Delete previous best model if exists
                if best_model_path is not None and best_model_path.exists():
                    best_model_path.unlink(missing_ok=True)
                
                if val_iou > self.save_threshold_iou:
                    # Save new best model to training_dir/models
                    iou_string = f'iou{val_iou:.4f}'.replace('.', '')
                    model_filename = f'ep{epoch+1:03d}_iou{iou_string}.pt'
                    best_model_path = self.training_dir / 'models' / model_filename
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Saved best model to {best_model_path}")
                
                # Save video if enabled
                if self.create_video and video_payload is not None:
                    try:
                        vgen = VideoGenerator(
                            video_payload["batch"],
                            video_payload["pred_masks"],
                            video_payload["true_masks"],
                            batch_idx=video_payload["batch_idx"],
                            output_dir=self.training_dir / 'videos'
                        )
                        video_path = vgen.save_sequence_gif(fps=10, alpha=0.5, frame_skip=1)
                        print(f"Saved video to {video_path}")
                    except Exception as e:
                        print(f"[ERROR][train] video generation failed:\n {e}")
            else:
                patience += 1
                if patience >= self.patience:
                    print("Early stopping")
                    break
        
        # Save training summary CSV at the end
        self._save_training_summary()
        
        return self
    
    def _save_training_summary(self):
        """Save training summary to CSV file"""
        if len(self.training_summary) > 0:
            df = pd.DataFrame(self.training_summary)
            summary_path = self.training_dir / 'training_summary.csv'
            df.to_csv(summary_path, index=False)
            print(f"Saved training summary to {summary_path}")
        else:
            print("Warning: No training summary to save")

    @staticmethod
    def _validate(model, loader, loss_fn, device, create_video: bool=True, max_batch_per_val:int = 1e3):
        model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_iou_orig_size = 0.0
        batches_per_val = 0
        video_payload = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batches_per_val += 1
                if batches_per_val > max_batch_per_val:
                    break
                frames = batch['frame'].to(device)
                target = batch['mask'].to(device)
                label_idx = batch['label_idx']
                pred = model(frames)  # (B, C, T, H, W)
                orig_shape = batch['orig_shape']
                orig_mask = batch['orig_mask'].to(device)

                loss = loss_fn(pred, target, label_idx)
                total_loss += loss.item()

                pred_probs = torch.sigmoid(slice_tensor_at_label(pred, label_idx))
                target = slice_tensor_at_label(target, label_idx)

                target_orig = slice_tensor_at_label(orig_mask, label_idx)
                prob_upsampled = torch.nn.functional.interpolate(pred_probs,  
                                                size=orig_shape, 
                                                mode='nearest')
                
                iou = SegmentationTrainer.compute_iou(pred_probs, target)
                iou_orig_size = SegmentationTrainer.compute_iou(prob_upsampled, target_orig)
                total_iou += iou.item()
                total_iou_orig_size += iou_orig_size.item()
                if create_video and video_payload is None:
                    # Capture first available batch for potential video generation
                    video_payload = {
                        "batch": batch,
                        "pred_masks": pred.cpu().numpy(),
                        "true_masks": target.cpu().numpy(),
                        "batch_idx": batch_idx,
                    }

        return total_loss / len(loader), total_iou / len(loader), total_iou_orig_size / len(loader), video_payload
