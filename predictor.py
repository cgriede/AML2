from utils import slice_tensor_at_label
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import csv
from multiprocessing import Pool
from tqdm import tqdm

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
    
    @staticmethod
    def predict_test(model, loader, device, seq_len: int = 16, num_postprocess_workers: int = None, test_data: list = None):
        """
        Generate predictions for test set - extract center frame from 3D output.
        Buffers probabilities per video, then postprocesses in parallel worker processes
        to keep GPU busy during postprocessing.
        
        Args:
            model: Trained model
            loader: DataLoader for test data
            device: Device to run inference on
            seq_len: Sequence length used by the model
            num_postprocess_workers: Number of worker processes for postprocessing (default: cpu_count - 1)
            test_data: Original test data list (optional, used to verify original video shapes)
        
        Returns:
            dict: Predictions in format {name: full_mask_array (H, W, T)}
        """
        if num_postprocess_workers is None:
            raise ValueError("num_postprocess_workers must be provided")
        
        model.eval()
        # Buffer predictions per video: {name: {frame_idx: {'probs': array, 'orig_shape': tuple}}}
        video_buffers = {}
        
        print("Generating predictions on GPU...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Predicting")):
                frames = batch['frame'].to(device)
                names = batch['name']
                frame_indices = batch['frame_idx'].cpu().numpy()
                label_indices = batch['label_idx']  # Frame index in sequence to extract (center frame)
                orig_shapes = batch.get('orig_shape', None)
                
                try:
                    pred = model(frames)  # Shape: (B, C, T, H, W)
                except Exception as e:
                    print(f"[ERROR][predict] model forward failed: {e}")
                    raise
                
                # Extract frame at label_idx from the sequence (same as validation)
                pred_center = slice_tensor_at_label(pred, label_indices)  # (B, C, H, W)
                pred_probs = torch.sigmoid(pred_center)  # Convert logits to probabilities
                
                # Move probabilities to CPU and store (don't convert to binary yet)
                pred_probs_cpu = pred_probs.cpu().numpy()  # (B, C, H, W)
                
                for i, name in enumerate(names):
                    if name not in video_buffers:
                        video_buffers[name] = {}
                    
                    frame_idx = int(frame_indices[i])
                    
                    # Handle orig_shape - convert to tuple if needed
                    if orig_shapes is not None:
                        orig_shape_raw = orig_shapes[i]
                        # Convert to tuple if it's a tensor or list
                        if isinstance(orig_shape_raw, (torch.Tensor, np.ndarray)):
                            orig_shape = tuple(orig_shape_raw.tolist() if hasattr(orig_shape_raw, 'tolist') else orig_shape_raw)
                        elif isinstance(orig_shape_raw, (list, tuple)):
                            orig_shape = tuple(orig_shape_raw)
                        else:
                            orig_shape = orig_shape_raw
                    else:
                        orig_shape = None
                    
                    # Store probabilities (not binary yet) - will postprocess later
                    # Extract probabilities: (C, H, W) -> (H, W)
                    probs = pred_probs_cpu[i, 0, :, :]  # Remove channel dimension
                    
                    video_buffers[name][frame_idx] = {
                        'probs': probs,
                        'orig_shape': orig_shape
                    }
        
        print(f"Postprocessing {len(video_buffers)} videos using {num_postprocess_workers} workers...")
        
        # Create a mapping from video name to original shape from test_data (if provided)
        test_data_shape_map = {}
        if test_data is not None:
            for item in test_data:
                name = item['name']
                # Get original video shape: (H, W, T) -> (H, W)
                orig_video_shape = item['video'].shape[:2]
                test_data_shape_map[name] = orig_video_shape
        
        # Prepare data for parallel postprocessing
        # Get orig_shape from first frame prediction for each video, but verify with test_data
        video_data_list = []
        for name, frame_predictions in video_buffers.items():
            # Get orig_shape from first frame (should be same for all frames of same video)
            first_frame_data = next(iter(frame_predictions.values()))
            orig_shape = first_frame_data['orig_shape']
            
            # Ensure orig_shape is a tuple of 2 integers
            if orig_shape is not None:
                if isinstance(orig_shape, (torch.Tensor, np.ndarray)):
                    orig_shape = tuple(orig_shape.tolist() if hasattr(orig_shape, 'tolist') else orig_shape)
                elif isinstance(orig_shape, (list, tuple)):
                    orig_shape = tuple(orig_shape)
            
            # Verify with test_data if available (this is the ground truth original shape)
            if name in test_data_shape_map:
                test_orig_shape = test_data_shape_map[name]
                # Use test_data shape as the authoritative source
                orig_shape = test_orig_shape
                print(f"Using original shape from test_data for {name}: {orig_shape}")
            elif orig_shape is None:
                # Fallback: use shape from probabilities (this means no upsampling needed)
                orig_shape = first_frame_data['probs'].shape
                print(f"Warning: No orig_shape found for {name}, using prob shape {orig_shape} (no upsampling)")
            else:
                # Check if orig_shape matches prob shape (means it's already resized)
                prob_shape = first_frame_data['probs'].shape
                if orig_shape == prob_shape:
                    print(f"Warning: orig_shape matches prob shape for {name}: {orig_shape} - this might be wrong!")
                    # If they match, it means orig_shape is actually the resized shape, not original
                    # We can't fix this without test_data, so warn the user
                else:
                    print(f"Using orig_shape from batch for {name}: {orig_shape} (prob shape: {prob_shape})")
            
            # Final validation: ensure we have (H, W)
            if not isinstance(orig_shape, tuple) or len(orig_shape) != 2:
                raise ValueError(f"Invalid orig_shape for video {name}: {orig_shape}, expected (H, W) tuple")
            
            video_data_list.append((name, frame_predictions, orig_shape))
        
        # Postprocess videos in parallel (using module-level function for multiprocessing)
        with Pool(processes=num_postprocess_workers) as pool:
            results = pool.map(_postprocess_video, video_data_list)
        
        # Convert results list to dictionary
        results_dict = {name: mask for name, mask in results if mask is not None}
        
        print(f"Postprocessing complete. Generated masks for {len(results_dict)} videos.")
        return results_dict
        
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
    def save_submission(predictions, test_data, expected_iou: float, output_dir: Path = None):
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
            output_dir = Path('results') / f"{timestamp}_{iou_str}"
        
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

            with open(output_dir / f"{name}.npy", 'wb') as f:
                np.save(f, mask)
            
            #NOTE: this might work if the new does not
            # Transpose to (T, H, W) first, then flatten row-major
            #flattened = np.transpose(mask, (2, 0, 1)).flatten(order='C')


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
