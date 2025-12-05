import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os


class VideoGenerator:
    """
    Generates videos from a sequence of valve images with labeled pixels highlighted in red.
    """
    def __init__(self, video_data, prediction_mask=None, output_dir="results/video"):
        """
        Args:
            video_data: Dictionary containing 'video', 'label', 'frames', 'name', 'dataset'
            prediction_mask: Optional predicted mask to compare with ground truth
            output_dir: Directory to save generated videos
        """
        self.video_data = video_data
        self.prediction_mask = prediction_mask
        self.output_dir = output_dir
        
        # Extract video information
        self.video_name = video_data['name']
        self.video = video_data['video']  # Shape: (height, width, num_frames)
        self.ground_truth = video_data['label']  # Ground truth mask
        self.labeled_frames = video_data['frames']  # List of frame indices with labels
        self.dataset = video_data['dataset']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_video(self, fps=20, show_prediction=False, alpha=0.5, frame_skip=1):
        """
        Generate and save video with labeled pixels highlighted in red.
        Saves as GIF (no ffmpeg dependency required).
        
        Args:
            fps: Frames per second for the output video
            show_prediction: If True and prediction_mask is provided, show prediction vs ground truth
            alpha: Transparency of the red overlay (0.0 to 1.0)
            frame_skip: Skip every N frames to reduce file size (1=all frames, 2=every other frame, etc)
        
        Returns:
            Path to the saved video file
        """
        # Create subdirectory for this video
        video_dir = os.path.join(self.output_dir, self.video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Set up figure
        if show_prediction and self.prediction_mask is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ax_gt, ax_pred = axes
            ax_gt.set_title("Ground Truth", fontsize=14, fontweight='bold')
            ax_pred.set_title("Prediction", fontsize=14, fontweight='bold')
            ax_gt.axis('off')
            ax_pred.axis('off')
            
            # Initialize images for ground truth
            im_video_gt = ax_gt.imshow(self.video[:, :, 0], cmap='gray', vmin=0, vmax=255)
            im_mask_gt = ax_gt.imshow(np.zeros_like(self.video[:, :, 0], dtype=float), 
                                       cmap='Reds', alpha=0.0, vmin=0, vmax=1)
            
            # Initialize images for prediction
            im_video_pred = ax_pred.imshow(self.video[:, :, 0], cmap='gray', vmin=0, vmax=255)
            im_mask_pred = ax_pred.imshow(np.zeros_like(self.video[:, :, 0], dtype=float), 
                                          cmap='Reds', alpha=0.0, vmin=0, vmax=1)
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.axis('off')
            
            # Initialize images
            im_video = ax.imshow(self.video[:, :, 0], cmap='gray', vmin=0, vmax=255)
            im_mask = ax.imshow(np.zeros_like(self.video[:, :, 0], dtype=float), 
                                cmap='Reds', alpha=0.0, vmin=0, vmax=1)
        
        # Update function for animation
        def update(frame_num):
            if show_prediction and self.prediction_mask is not None:
                # Update both ground truth and prediction
                im_video_gt.set_data(self.video[:, :, frame_num])
                im_video_pred.set_data(self.video[:, :, frame_num])
                
                if frame_num in self.labeled_frames:
                    # Ground truth mask
                    mask_gt = self.ground_truth[:, :, frame_num].astype(float)
                    im_mask_gt.set_data(mask_gt)
                    im_mask_gt.set_alpha(alpha)
                    
                    # Prediction mask
                    mask_pred = self.prediction_mask[:, :, frame_num].astype(float)
                    im_mask_pred.set_data(mask_pred)
                    im_mask_pred.set_alpha(alpha)
                    
                    # Count pixels for stats
                    gt_pixels = mask_gt.sum()
                    pred_pixels = mask_pred.sum()
                    ax_gt.set_title(f"Ground Truth - Frame {frame_num}\n{int(gt_pixels)} pixels", 
                                   fontsize=12, color='red')
                    ax_pred.set_title(f"Prediction - Frame {frame_num}\n{int(pred_pixels)} pixels", 
                                     fontsize=12, color='red')
                else:
                    im_mask_gt.set_alpha(0.0)
                    im_mask_pred.set_alpha(0.0)
                    ax_gt.set_title("Ground Truth", fontsize=14, fontweight='bold')
                    ax_pred.set_title("Prediction", fontsize=14, fontweight='bold')
                
                return [im_video_gt, im_mask_gt, im_video_pred, im_mask_pred]
            else:
                # Single view with ground truth only
                im_video.set_data(self.video[:, :, frame_num])
                
                if frame_num in self.labeled_frames:
                    mask = self.ground_truth[:, :, frame_num].astype(float)
                    im_mask.set_data(mask)
                    im_mask.set_alpha(alpha)
                    
                    pixels = mask.sum()
                    ax.set_title(f"Frame {frame_num} - {int(pixels)} labeled pixels", 
                               fontsize=14, color='red', fontweight='bold')
                else:
                    im_mask.set_alpha(0.0)
                    ax.set_title(f"Frame {frame_num}", fontsize=14, fontweight='bold')
                
                return [im_video, im_mask]
        
        # Create frame list with skip
        frame_list = list(range(0, self.video.shape[2], frame_skip))
        
        # Create animation
        interval = (1000 // fps) * frame_skip  # milliseconds per frame
        ani = FuncAnimation(fig, update, frames=frame_list, 
                          interval=interval, blit=True, repeat=True)
        
        # Save video as GIF
        output_filename = f"{self.video_name}_labeled.gif"
        if show_prediction:
            output_filename = f"{self.video_name}_comparison.gif"
        
        output_path = os.path.join(video_dir, output_filename)
        
        try:
            # Use PillowWriter for GIF (built-in, no external dependencies)
            writer = PillowWriter(fps=max(fps // frame_skip, 1))
            ani.save(output_path, writer=writer)
            print(f"✓ Video saved to: {output_path}")
        except Exception as e:
            print(f"✗ Error saving video: {e}")
            print(f"  Attempting to save individual frames instead...")
            output_path = self._save_frames_as_images(video_dir, frame_list, update, show_prediction)
        
        plt.close(fig)
        
        print(f"  Total frames: {self.video.shape[2]}")
        print(f"  Labeled frames: {len(self.labeled_frames)}")
        print(f"  Output frames (with skip={frame_skip}): {len(frame_list)}")
        
        return output_path
    
    def _save_frames_as_images(self, video_dir, frame_list, update_func, show_prediction):
        """Fallback: Save individual frames as PNG images"""
        frames_dir = os.path.join(video_dir, f"{self.video_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"  Saving {len(frame_list)} frames to {frames_dir}...")
        
        for i, frame_num in enumerate(frame_list):
            if i % max(1, len(frame_list) // 10) == 0:
                print(f"    {i+1}/{len(frame_list)}")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.axis('off')
            
            frame_img = self.video[:, :, frame_num]
            ax.imshow(frame_img, cmap='gray', vmin=0, vmax=255)
            
            # Overlay mask if labeled
            if frame_num in self.labeled_frames:
                mask = self.ground_truth[:, :, frame_num].astype(float)
                ax.imshow(mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                ax.set_title(f"Frame {frame_num} - {int(mask.sum())} pixels (LABELED)", 
                           fontsize=12, color='red')
            else:
                ax.set_title(f"Frame {frame_num}", fontsize=12)
            
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:04d}.png")
            plt.savefig(frame_path, bbox_inches='tight', dpi=80)
            plt.close(fig)
        
        print(f"✓ Frames saved to: {frames_dir}")
        return frames_dir
    
    def print_stats(self):
        """Print statistics about the video and labeled pixels."""
        print(f"\n{'='*70}")
        print(f"Video: {self.video_name} (Dataset: {self.dataset})")
        print(f"{'='*70}")
        print(f"Total frames: {self.video.shape[2]}")
        print(f"Video dimensions: {self.video.shape[0]}×{self.video.shape[1]} pixels")
        print(f"Labeled frames: {self.labeled_frames}")
        print(f"\nPixel statistics for labeled frames:")
        
        total_pixels = 0
        for f in self.labeled_frames:
            mask_sum = self.ground_truth[:, :, f].sum()
            percentage = (mask_sum / (self.video.shape[0] * self.video.shape[1])) * 100
            print(f"  Frame {f:3d}: {int(mask_sum):6d} pixels ({percentage:5.2f}% of frame)")
            total_pixels += mask_sum
        
        avg_pixels = total_pixels / len(self.labeled_frames) if self.labeled_frames else 0
        print(f"\n  Average: {int(avg_pixels):6d} pixels per labeled frame")
        
        if self.prediction_mask is not None:
            print(f"\n{'─'*70}")
            print(f"Prediction statistics:")
            print(f"{'─'*70}")
            
            total_recall = 0
            total_precision = 0
            count = 0
            
            for f in self.labeled_frames:
                pred_sum = self.prediction_mask[:, :, f].sum()
                gt_sum = self.ground_truth[:, :, f].sum()
                overlap = (self.prediction_mask[:, :, f] & self.ground_truth[:, :, f]).sum()
                
                if gt_sum > 0:
                    recall = (overlap / gt_sum) * 100
                    precision = (overlap / pred_sum) * 100 if pred_sum > 0 else 0.0
                    total_recall += recall
                    total_precision += precision
                    count += 1
                    
                    print(f"  Frame {f:3d}: Pred={int(pred_sum):6d}, GT={int(gt_sum):6d}, "
                          f"Overlap={int(overlap):6d} | Recall: {recall:5.1f}%, Precision: {precision:5.1f}%")
            
            if count > 0:
                avg_recall = total_recall / count
                avg_precision = total_precision / count
                print(f"\n  Average Recall: {avg_recall:5.1f}%, Average Precision: {avg_precision:5.1f}%")
        
        print(f"{'='*70}\n")