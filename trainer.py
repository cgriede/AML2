from utils import slice_tensor_at_label
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
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

        #if there is only one frame, return 0
        if pred.shape[2] < 2:
            return torch.tensor(0.0)
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
        #if there is only one frame, return 0
        if probs.shape[2] < 2:
            return torch.tensor(0.0)
        # Sum foreground probability mass per frame in batch
        areas = probs.sum(dim=[2,3,4])  # → (B,)
        # Penalize large changes between consecutive frames in batch (approximation)
        if areas.shape[0] > 1:
            diff = torch.abs(areas[1:] - areas[:-1])
            return self.weight * diff.mean()
        return 0.0 * areas.sum()  # 0 if batch=1


class SegmentationTrainer:

    allow_no_user_input: bool = True

    class CombinedLoss(nn.Module):
        def __init__(self,
                    dice_weight: float = 50,
                    bce_weight: float = 50,
                    tsl_weight: float = 1,
                    ac_weight: float = 0.5) -> None:
            super().__init__()
            self.dice = BinaryDiceLoss()
            self.bce = nn.BCEWithLogitsLoss()
            self.tsl = TemporalSmoothLoss()
            self.ac = AreaConsistencyLoss()

            total = dice_weight + bce_weight + tsl_weight + ac_weight
            self.dice_weight = dice_weight / total
            self.bce_weight = bce_weight / total
            self.tsl_weight = tsl_weight / total
            self.ac_weight = ac_weight / total

        @staticmethod
        def _get_downsampled_pred_target_pairs(target: torch.Tensor, preds: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
            pred_target_pairs = []
            for i, pred in enumerate(preds):
                # Downsample target to pred size (nearest to keep binary)
                target_resized = torch.nn.functional.interpolate(
                    target.float(), size=pred.shape[2:], mode='nearest'
                ).long()  # or .bool() if binary
                pred_target_pairs.append((pred, target_resized))
            return pred_target_pairs

        def forward(self, logits, target_original, label_idx):
            """
            logits: either single tensor or list of tensors (deep supervision)
            target: full mask sequence (B, 1, T, H, W) or (B, T, H, W)
            label_idx: list of lists — original labeled frame indices per video in batch
            """
            # Handle deep supervision: logits can be list
            if isinstance(logits, list):
                pred_target_pairs = self._get_downsampled_pred_target_pairs(target_original, logits)
            else:
                pred_target_pairs = [(logits, target_original)]

            total_loss = 0.0
            weights = [1.0, 0.5, 0.25, 0.125, 0.0625]  # fine to coarse

            for (pred, target), w in zip(pred_target_pairs, weights):
                # Compute temporal downsampling factor for this head
                # pred.shape[2] = current T, target.shape[2] = original T
                orig_T = target.shape[2]
                current_T = pred.shape[2]
                stride_t = orig_T // current_T  # e.g., 1, 2, 4, 8, 16

                # Scale label indices to this resolution
                # Scale label indices to this resolution
                scaled_label_idx = []
                for idx in label_idx:  # idx_tensor is always 1D tensor (shape (N,) or (1,))  # converts to Python list of ints, works for both [8] and [41,65,167]
                    scaled = [idx // stride_t]
                    scaled_label_idx.append(scaled)

                # Only compute Dice/BCE on labeled frames at this scale
                if any(len(s) > 0 for s in scaled_label_idx):  # if any video has labels
                    pred_label = slice_tensor_at_label(pred, scaled_label_idx)
                    mask_label = slice_tensor_at_label(target, scaled_label_idx)

                    probs_label = torch.sigmoid(pred_label)
                    dice_loss = self.dice(probs_label, mask_label)
                    bce_loss = self.bce(pred_label, mask_label.float())
                else:
                    dice_loss = bce_loss = torch.tensor(0.0, device=pred.device)

                # Temporal smoothness and area consistency on full-volume probs
                probs = torch.sigmoid(pred)  # full spatio-temporal volume
                temp_loss = self.tsl(probs)
                ac_loss = self.ac(probs)

                head_loss = (self.dice_weight * dice_loss +
                            self.bce_weight * bce_loss +
                            self.tsl_weight * temp_loss +
                            self.ac_weight * ac_loss)

                total_loss += w * head_loss

                if torch.isnan(head_loss).item():
                    raise ValueError(f"Head loss is NaN for head {w}")

            return total_loss

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
                deep_supervision: bool = False,
                description: str = None,
                no_validation: bool = False,
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
        self.deep_supervision = deep_supervision
        self.description = description
        self.best_model_path = None
        self.no_validation = no_validation
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
        
        if not self.allow_no_user_input and self.description is None:
            self._get_user_training_description()
        elif self.description is not None:
            self._get_user_training_description(self.description)
        else:
            print("Allowing no user input, continuing without user input")
        
    def _get_user_training_description(self, description: str = None):
        if description is None:
            description = input("Enter a description for the training: ")
        with open(self.training_dir / 'description.txt', 'w') as f:
            f.write(description)
        print(f"Training description saved to {self.training_dir / 'description.txt'}")
        return description

    def _train_epoch(self,model, loader, optimizer, loss_fn, device, max_batch_per_epoch: int = 1e3):
        model.train()
        total_loss = 0.0
        batches_per_epoch = 0
        train_iou = 0.0
        
        for batch_idx, batch in enumerate(loader):
            batches_per_epoch += 1
            if batches_per_epoch > max_batch_per_epoch:
                break
            frames = batch['frame'].to(device)
            target = batch['mask'].to(device)
            label_idx = batch['label_idx'].to(device)
            optimizer.zero_grad()
            logits = model(frames)
            loss = loss_fn(logits, target, label_idx)
            # Single backward on the full weighted loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            with torch.no_grad():
                if self.deep_supervision:
                    logits = logits[0]
                pred_probs = torch.sigmoid(slice_tensor_at_label(logits, label_idx))
                target = slice_tensor_at_label(target, label_idx)
                train_iou += self.compute_iou(pred_probs, target).item()
        return total_loss / len(loader), train_iou / len(loader)

    def train_model(self):
        best_val_iou = 0.0
        best_train_iou = 0.0
        patience = 0
        for epoch in range(self.n_epochs):
            train_loss, train_iou = self._train_epoch(self.model, self.train_loader, self.optimizer, self.loss_fn, self.device, self.max_batch_per_epoch)
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train IoU={train_iou:.4f}")
            if not self.no_validation:
                val_loss, val_iou, val_iou_orig, median_iou, median_iou_orig, video_payload = self._validate(self.model, self.val_loader, self.loss_fn, self.device, self.create_video, self.max_batch_per_val)
                print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, \n\tVal IoU (mean)={val_iou:.4f}, \n\tVal IoU (orig, mean)={val_iou_orig:.4f}, \n\tVal IoU (median, Kaggle-style)={median_iou:.4f}, \n\tVal IoU (orig, median)={median_iou_orig:.4f}")
                # Record training summary
                self.training_summary.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_iou': train_iou,
                    'val_loss': val_loss,
                    'val_iou': val_iou,
                    'val_iou_orig': val_iou_orig,
                    'median_iou': median_iou,
                    'median_iou_orig': median_iou_orig
                })
            else:

                self.training_summary.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_iou': train_iou,
                })
            
            if val_iou > best_val_iou or train_iou > best_train_iou:
                if self.no_validation:
                    best_train_iou = train_iou
                else:
                    best_val_iou = val_iou
                patience = 0
                # Delete previous best model if exists
                if self.best_model_path is not None and self.best_model_path.exists():
                    self.best_model_path.unlink(missing_ok=True)
                
                if max(best_val_iou, best_train_iou) > self.save_threshold_iou:
                    # Save new best model to training_dir/models
                    iou_string = f'iou{max(best_val_iou, best_train_iou):.4f}'.replace('.', '')
                    model_filename = f'ep{epoch+1:03d}_iou{iou_string}.pt'
                    self.best_model_path = self.training_dir / 'models' / model_filename
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f"Saved best model to {self.best_model_path}")
                
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

    def _validate(self, model, loader, loss_fn, device, create_video: bool=True, max_batch_per_val:int = 1e3):
        model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_iou_orig_size = 0.0
        batches_per_val = 0
        video_payload = None
        
        # Collect IoU per video to compute median (matching Kaggle's evaluation)
        iou_per_video = {}  # {video_name: [iou_values]}
        iou_orig_per_video = {}  # {video_name: [iou_values]}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batches_per_val += 1
                if batches_per_val > max_batch_per_val:
                    break
                frames = batch['frame'].to(device)
                target = batch['mask'].to(device)
                label_idx = batch['label_idx']
                pred = model(frames)  # (B, C, T, H, W)
                if self.deep_supervision: #only take the finest prediction for validation
                    pred = pred[0]
                orig_shape = batch['orig_shape'][0].item(), batch['orig_shape'][1].item()
                orig_mask = batch['orig_mask'].to(device)
                video_names = batch.get('video_name', [f'batch_{batch_idx}'] * frames.size(0))

                loss = loss_fn(pred, target, label_idx)
                total_loss += loss.item()

                pred_probs = torch.sigmoid(slice_tensor_at_label(pred, label_idx))
                target = slice_tensor_at_label(target, label_idx)

                target_orig = slice_tensor_at_label(orig_mask, label_idx)
                prob_upsampled = torch.nn.functional.interpolate(pred_probs,  
                                                size=(1,*orig_shape), 
                                                mode='nearest')
                
                iou = SegmentationTrainer.compute_iou(pred_probs, target)
                iou_orig_size = SegmentationTrainer.compute_iou(prob_upsampled, target_orig)
                total_iou += iou.item()
                total_iou_orig_size += iou_orig_size.item()
                
                # Track IoU per video (handle batch dimension)
                batch_size = frames.size(0)
                if isinstance(video_names, (list, tuple)) and len(video_names) == batch_size:
                    for i in range(batch_size):
                        vname = video_names[i]
                        if vname not in iou_per_video:
                            iou_per_video[vname] = []
                            iou_orig_per_video[vname] = []
                        # For batched predictions, we compute IoU per sample
                        # Note: compute_iou aggregates over batch, so we need per-sample IoU
                        # For now, we'll use the batch IoU and track it per video
                        # This is approximate - ideally we'd compute per-sample IoU
                        iou_per_video[vname].append(iou.item())
                        iou_orig_per_video[vname].append(iou_orig_size.item())
                
                if create_video and video_payload is None:
                    # Capture first available batch for potential video generation
                    video_payload = {
                        "batch": batch,
                        "pred_masks": pred.cpu().numpy(),
                        "true_masks": target.cpu().numpy(),
                        "batch_idx": batch_idx,
                    }

        # Compute mean IoU (current method)
        mean_iou = total_iou / len(loader)
        mean_iou_orig = total_iou_orig_size / len(loader)
        
        # Compute median IoU per video (matching Kaggle's evaluation)
        # Take median of per-video mean IoU values
        if iou_per_video:
            video_mean_ious = [np.mean(iou_list) for iou_list in iou_per_video.values()]
            video_mean_ious_orig = [np.mean(iou_list) for iou_list in iou_orig_per_video.values()]
            median_iou = np.median(video_mean_ious)
            median_iou_orig = np.median(video_mean_ious_orig)
        else:
            median_iou = mean_iou
            median_iou_orig = mean_iou_orig

        return total_loss / len(loader), mean_iou, mean_iou_orig, median_iou, median_iou_orig, video_payload
