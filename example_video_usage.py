"""
Example usage of the VideoGenerator class.
Shows how to generate videos with labeled pixels highlighted in red.
"""

import pickle
import gzip
from video_generator import VideoGenerator

# Load your data
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

# Load train data
print("Loading train data...")
train_data = load_zipped_pickle("train.pkl")
print(f"✓ Loaded {len(train_data)} videos")

# Example 1: Generate video for first expert dataset video
print("\n" + "="*60)
print("Example 1: Generate labeled video (ground truth only)")
print("="*60)

expert_data = None
for item in train_data:
    if item['dataset'] == 'expert':
        expert_data = item
        break

if expert_data:
    # Create video generator
    vg = VideoGenerator(expert_data, output_dir="results/video")
    
    # Print statistics
    vg.print_stats()
    
    # Generate video with labeled pixels in red
    video_path = vg.generate_video(fps=20, alpha=0.5)
    print(f"\n✓ Video generated successfully!")


# Example 2: Generate comparison video with predictions
print("\n" + "="*60)
print("Example 2: Generate comparison video (ground truth vs prediction)")
print("="*60)

# Assume you have a prediction mask from your model
# For demonstration, let's create a dummy prediction (replace with your actual model output)
import numpy as np

if expert_data:
    # Create a dummy prediction mask (replace this with your actual model predictions)
    # This example just shifts the ground truth slightly to show the comparison
    prediction_mask = np.zeros_like(expert_data['label'], dtype=bool)
    
    # Copy ground truth and add some noise for demonstration
    for frame_idx in expert_data['frames']:
        gt = expert_data['label'][:, :, frame_idx]
        # Shift and add some random pixels to simulate prediction errors
        prediction_mask[:, :, frame_idx] = np.roll(gt, shift=(2, 2), axis=(0, 1))
    
    # Create video generator with prediction
    vg_compare = VideoGenerator(expert_data, 
                                prediction_mask=prediction_mask,
                                output_dir="results/video")
    
    # Print statistics (includes prediction metrics)
    vg_compare.print_stats()
    
    # Generate comparison video
    video_path = vg_compare.generate_video(fps=20, show_prediction=True, alpha=0.5)
    print(f"\n✓ Comparison video generated successfully!")


# Example 3: Process multiple videos
print("\n" + "="*60)
print("Example 3: Process all expert videos")
print("="*60)

expert_videos = [item for item in train_data if item['dataset'] == 'expert']
print(f"Found {len(expert_videos)} expert videos")

for i, video_data in enumerate(expert_videos[:3]):  # Process first 3 for demo
    print(f"\nProcessing video {i+1}/{min(3, len(expert_videos))}: {video_data['name']}")
    vg = VideoGenerator(video_data, output_dir="results/video")
    vg.generate_video(fps=20, alpha=0.5)

print("\n✓ All videos processed!")
print(f"Check the 'results/video/' directory for output files.")
