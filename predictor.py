from utils import slice_tensor_at_label
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import csv
from multiprocessing import Pool

def _postprocess_video(video_data):
    """
    Postprocess a single video's predictions: convert probabilities to binary masks,
    upsample to original resolution, and create full video mask.
    Module-level function for multiprocessing compatibility.
    
    Args:
        video_data: Tuple of (name, frame_predictions_dict, orig_shape)
    
    Returns:
        Tuple of (name, full_mask_array)
    """
    name, frame_predictions, orig_shape = video_data
    
    if not frame_predictions:
        return name, None
    
    # Ensure orig_shape is a tuple of 2 integers
    if isinstance(orig_shape, (torch.Tensor, np.ndarray)):
        orig_shape = tuple(orig_shape.tolist() if hasattr(orig_shape, 'tolist') else orig_shape)
    elif isinstance(orig_shape, (list, tuple)):
        orig_shape = tuple(orig_shape)
    
    # Validate orig_shape format
    if not isinstance(orig_shape, tuple) or len(orig_shape) != 2:
        # Fallback: use shape from first frame's probabilities
        first_frame_data = next(iter(frame_predictions.values()))
        orig_shape = first_frame_data['probs'].shape
        if len(orig_shape) != 2:
            raise ValueError(f"Invalid orig_shape for video {name}: {orig_shape}, expected (H, W) tuple")
    
    # Get max frame index to determine video length
    max_frame_idx = max(frame_predictions.keys())
    orig_h, orig_w = int(orig_shape[0]), int(orig_shape[1])
    
    # Create full mask array (H, W, T) where T = max_frame_idx + 1
    num_frames = max_frame_idx + 1
    full_mask = np.zeros((orig_h, orig_w, num_frames), dtype=bool)
    
    for frame_idx, pred_data in frame_predictions.items():
        # pred_data contains 'probs' (probabilities) at resized resolution
        probs_resized = pred_data['probs']  # (H_resized, W_resized) probabilities
        
        # Upsample probabilities to original resolution FIRST (same as validation)
        # Use the same method as validation: torch.nn.functional.interpolate with mode='nearest'
        if probs_resized.shape != (orig_h, orig_w):
            # Convert to torch tensor for interpolation (matching validation method)
            # interpolate expects (N, C, H, W), so add batch and channel dims
            probs_tensor = torch.from_numpy(probs_resized)[None, None, :, :]  # (1, 1, H, W)
            probs_upsampled = torch.nn.functional.interpolate(
                probs_tensor,
                size=(orig_h, orig_w),
                mode='nearest'
            )
            probs_orig = probs_upsampled[0, 0, :, :].numpy()  # (orig_h, orig_w)
        else:
            probs_orig = probs_resized
        
        # THEN apply binary threshold (same as validation: after upsampling)
        mask_orig = (probs_orig > 0.5).astype(bool)
        
        # Store in correct frame position
        if 0 <= frame_idx < num_frames:
            full_mask[:, :, frame_idx] = mask_orig
    
    return name, full_mask


class SegmentationPredictor:
    """Predictor class for generating test predictions and submission files"""

    def __init__(self, model, loader, device, seq_len: int = 16, num_postprocess_workers: int = None, test_data: list = None, deep_supervision: bool = False, probability_threshold: float = 0.5):
        self.model = model
        self.loader = loader
        self.device = device
        self.seq_len = seq_len
        self.num_postprocess_workers = num_postprocess_workers
        self.test_data = test_data
        self.deep_supervision = deep_supervision
        self.probability_threshold = probability_threshold
    
    def predict_test(self):
        """
        Generate predictions for the test set.
        - Model outputs (B, 1, T, H, W)
        - We extract the center frame (defined by label_idx)
        - Upsample to original spatial resolution if needed
        - Reconstruct full-video mask with predictions only at center frames
        """
        self.model.eval()
        predictions = {}  # {video_name: np.array (H_orig, W_orig, T)}
        batch_idx = 0
        disp_int = len(self.loader)/10
        disp_int = int(disp_int)
        with torch.no_grad():
            for batch in self.loader:
                batch_idx += 1
                if batch_idx % disp_int == 0:
                    print(f"Predicting batch {batch_idx} of {len(self.loader)} ({batch_idx/len(self.loader)*100:.2f}%)")
                frames = batch['frame'].to(self.device)           # (B, C, T, H, W)
                names = batch['name']                             # list of str
                frame_indices = batch['frame_idx'].cpu().numpy() # global frame idx in video
                label_indices = batch['label_idx']                # center frame idx in sequence
                orig_shapes = batch.get('orig_shape')             # list of (H_orig, W_orig) or None
                video_lengths = batch.get('video_length')          # list of int

                # Forward pass
                logits = self.model(frames)                       # may be list if deep_supervision
                if self.deep_supervision:
                    logits = logits[0]                            # use finest head

                # Extract center frame prediction (B, 1, H, W)
                center_logits = slice_tensor_at_label(logits, label_indices)
                center_probs = torch.sigmoid(center_logits)  # (B, 1, 1, H, W)

                # Process each video in batch
                for i, name in enumerate(names):
                    prob = center_probs[i]# (1, 1, H, W)
                    H_orig, W_orig = orig_shapes[0], orig_shapes[1]
                    T = video_lengths[i]
                    # Upsample to original resolution if needed
                    if (H_orig, W_orig) != prob.shape:
                        prob = torch.nn.functional.interpolate(
                            prob,
                            size=(H_orig, W_orig),
                            mode='nearest'
                        ).cpu().numpy() # (H, W)

                    global_frame_idx = int(frame_indices[i])

                    # Initialize full mask if first time seeing this video
                    if name not in predictions:
                        # Get total frames T from test_data or infer from max index + 1
                        predictions[name] = np.zeros((H_orig, W_orig, T), dtype=np.bool_)

                    # Place prediction at correct global frame
                    predictions[name][:, :, global_frame_idx] = (prob > self.probability_threshold)


        return predictions
        
    @staticmethod
    def get_sequences(arr):
        """
        Exact implementation from task3.ipynb notebook.
        Finds sequences of consecutive 1s in a boolean array.
        
        Args:
            arr: Boolean or integer array (0s and 1s)
        
        Returns:
            first_indices: List of start indices for each run
            lengths: List of lengths for each run
        """
        first_indices, last_indices, lengths = [], [], []
        n, i = len(arr), 0
        arr = [0] + list(arr) + [0]
        for index, value in enumerate(arr[:-1]):
            if arr[index+1]-arr[index] == 1:
                first_indices.append(index)
            if arr[index+1]-arr[index] == -1:
                last_indices.append(index)
        lengths = list(np.array(last_indices)-np.array(first_indices))
        return first_indices, lengths

    @staticmethod
    def save_submission(predictions, test_data, expected_iou: float, path_notes:str = "",  output_dir: Path = None):
        """
        Save predictions as CSV submission file in the format of sample.csv.
        Format: id,value where value is RLE encoding as string "[start, length]"
        
        Args:
            predictions: Dict from predict_test {name: mask_array (H, W, T)}
            test_data: Original test data list
            expected_iou: Expected IoU for directory naming
            output_dir: Output directory (default: results/YMD_{expected_iou})
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            iou_str = f"{expected_iou:.4f}".replace('.', '')
            output_dir = Path('results') / f"{timestamp}_{iou_str}_{path_notes}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'submission.csv'
        
        rows = []
        
        # Process each video in test_data
        for item in test_data:
            name = item['name']
            if name not in predictions:
                print(f"Warning: No predictions found for {name}")
                continue
            
            mask = predictions[name]  # (H, W, T) boolean array

            # Flatten entire video mask: (H, W, T) -> (H*W*T,) in row-major order
            # This flattens frame by frame: all pixels of frame 0, then frame 1, etc.
            # Within each frame, it's row-major (C-style): row 0, row 1, etc.
            flattened = mask.flatten(order='C')  # (H*W*T,) - row-major order to match sample.csv
            
            # Use the EXACT get_sequences function from the notebook
            # Convert boolean array to list (get_sequences expects a list)
            arr_list = flattened.astype(int).tolist()
            first_indices, lengths = SegmentationPredictor.get_sequences(arr_list)
            
            # Create one row per run
            # IMPORTANT: first_indices from get_sequences are already correct - NO offset adjustment!
            for run_idx, (start, length) in enumerate(zip(first_indices, lengths)):
                rows.append({
                    'id': f"{name}_{run_idx}",
                    'value': f"[{start}, {length}]"
                })
        
        # Save to CSV with explicit parameters to match sample.csv format exactly
        # Note: sample.csv has id unquoted and value quoted, so we use QUOTE_MINIMAL
        df = pd.DataFrame(rows)
        df.to_csv(
            output_file, 
            index=False,
            quoting=csv.QUOTE_MINIMAL,  # Only quote when necessary (value strings will be quoted, id won't)
            encoding='utf-8'  # Explicit UTF-8 encoding
        )
        print(f"Submission saved to {output_file}")
        print(f"Total rows: {len(rows)}")
        
        return output_file, output_dir
    
    @staticmethod
    def generate_full_videos(predictions, test_data, output_dir: Path, max_videos: int = 20):
        """
        Generate visualization videos for full-length predictions.
        
        Args:
            predictions: Dict from predict_test {name: full_mask_array (H, W, T)}
            test_data: Original test data list (contains full videos)
            output_dir: Directory to save videos (will create 'videos' subdirectory)
            max_videos: Maximum number of videos to generate
        """
        from video_generator import VideoGenerator
        
        video_dir = output_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating full-length videos for up to {max_videos} videos...")
        
        video_count = 0
        for item in test_data:
            if video_count >= max_videos:
                break
                
            name = item['name']
            if name not in predictions:
                print(f"Warning: No predictions found for {name}, skipping video")
                continue
            
            try:
                # Get full video frames: (H, W, T) -> normalize to [0, 1]
                full_video = item['video'].astype(np.float32) / 255.0  # (H, W, T)
                H, W, T = full_video.shape
                
                # Get full predictions: (H, W, T) -> (T, H, W) for VideoGenerator
                full_mask = predictions[name]  # (H, W, T) boolean
                pred_mask_hwt = full_mask.astype(np.float32)  # (H, W, T)
                
                # Ensure predictions and video have matching spatial dimensions
                if pred_mask_hwt.shape[:2] != (H, W):
                    print(f"Warning: Shape mismatch for {name}: video {full_video.shape} vs mask {pred_mask_hwt.shape}")
                    print(f"  Resizing mask from {pred_mask_hwt.shape[:2]} to {(H, W)}")
                    # Resize mask to match video dimensions using nearest neighbor
                    from data_prep import resize_frame
                    pred_mask_resized = np.zeros((H, W, T), dtype=pred_mask_hwt.dtype)
                    for t in range(T):
                        pred_mask_resized[:, :, t] = resize_frame(
                            pred_mask_hwt[:, :, t], 
                            (H, W), 
                            method='nearest'
                        )
                    pred_mask_hwt = pred_mask_resized
                
                # Transpose to (T, H, W) for VideoGenerator
                video_frames_thw = np.transpose(full_video, (2, 0, 1))  # (T, H, W)
                pred_mask_thw = np.transpose(pred_mask_hwt, (2, 0, 1))  # (T, H, W)
                
                # Verify shapes match
                assert video_frames_thw.shape == pred_mask_thw.shape, \
                    f"Shape mismatch: frames {video_frames_thw.shape} vs mask {pred_mask_thw.shape}"
                
                # Create dummy batch dict for VideoGenerator
                # VideoGenerator expects batch['frame'] with shape (B, 1, T, H, W) or (B, T, H, W)
                batch_frames = video_frames_thw[None, None, :, :, :]  # (1, 1, T, H, W)
                
                batch_dict = {
                    'frame': batch_frames,
                    'name': [name],  # List with single name
                    'video_name': [name]  # Alternative name field
                }
                
                # Create dummy true masks (zeros) since test data has no ground truth
                true_mask_thw = np.zeros_like(pred_mask_thw)  # (T, H, W)
                
                # Reshape for VideoGenerator: (T, H, W) -> (1, 1, T, H, W) for batch
                pred_masks_batch = pred_mask_thw[None, None, :, :, :]  # (1, 1, T, H, W)
                true_masks_batch = true_mask_thw[None, None, :, :, :]  # (1, 1, T, H, W)
                
                # Generate video
                vgen = VideoGenerator(
                    batch_dict,
                    pred_masks_batch,  # (1, 1, T, H, W)
                    true_masks_batch,  # (1, 1, T, H, W) - all zeros (no ground truth)
                    batch_idx=0,
                    output_dir=str(video_dir)
                )
                video_path = vgen.save_sequence_gif(fps=10, alpha=0.5, frame_skip=1)
                print(f"Saved full-length video for {name} ({T} frames) to {video_path}")
                video_count += 1
                
            except Exception as e:
                print(f"[ERROR] Video generation failed for {name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Generated {video_count} full-length prediction videos in {video_dir}")
