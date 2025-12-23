"""
Data loading and preprocessing for Mitral Valve Segmentation
"""

import numpy as np
import torch
import gzip
import pickle
from torch.utils.data import Dataset, ConcatDataset
from scipy import ndimage
from enum import Enum
import random

def load_zipped_pickle(filename):
    """Load a pickle file, handling both plain .pkl and gzipped .pkl.gz"""
    with open(filename, 'rb') as f:
        # Peek at first 2 bytes to check for gzip magic
        magic = f.read(2)
        f.seek(0)  # Reset file pointer
        
        if magic == b'\x1f\x8b':  # Gzip magic bytes
            print(f"{filename} is gzipped – decompressing...")
            with gzip.open(filename, 'rb') as gf:
                return pickle.load(gf)
        else:
            print(f"{filename} is plain pickle (starts with {magic})")
            return pickle.load(f)


def resize_frame(frame, target_size=(128, 128), method='linear'):
    """
    Resize frame using scipy zoom.
    
    Args:
        frame: Input array
        target_size: Target (height, width)
        method: 'linear' for bilinear interpolation, 'nearest' for nearest neighbor
    """
    h, w = frame.shape[:2]
    if (h, w) == target_size:
        return frame
    
    # Calculate zoom factors
    zoom_h = target_size[0] / h
    zoom_w = target_size[1] / w
    
    # order: 0=nearest, 1=linear, 3=cubic
    order = 0 if method == 'nearest' else 1
    resized = ndimage.zoom(frame, (zoom_h, zoom_w) + (1,) * (frame.ndim - 2), order=order)
    return resized


class DatasetType(Enum):
    EXPERT = "expert"
    AMATEUR = "amateur"
    MIXED = "mixed"


class MitralValveDataset(Dataset):
    """
    Load temporal sequences from train.pkl or test.pkl - lazy loading, compute on demand.
    
    Modes:
    - "train": Apply data augmentation, return resized data for training
    - "val": No augmentation, return resized + original data for accurate IoU calculation
    - "test": Process all frames (no labels), return data for test predictions
    """
    
    def __init__(self, data_list, mode: str = "train",
                 seq_len: int=16,
                 random_label_position: bool=False,
                 rotation_chance: float=0.0,
                 rotation_angle: float=20,
                 target_size: tuple=None,
                 dataset_type: DatasetType=DatasetType.EXPERT):
        """
        Args:
            data_list: List of data items from train.pkl or test.pkl
            mode: "train", "val", or "test"
            seq_len: Sequence length (number of frames)
            random_label_position: Whether to randomize label position in sequence (train only)
            rotation_chance: Probability of applying rotation augmentation (train only)
            rotation_angle: Maximum rotation angle in degrees (train only)
            target_size: Target (height, width) for resizing
            dataset_type: Filter by dataset type (EXPERT, AMATEUR, MIXED)
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"mode must be 'train', 'val', or 'test', got '{mode}'")
        
        self.data_list = data_list
        self.mode = mode
        self.seq_len = seq_len
        self.random_label_position = random_label_position if mode == "train" else False
        self.rotation_chance = rotation_chance if mode == "train" else 0.0
        self.rotation_angle = rotation_angle
        self.target_size = target_size
        self.dataset_type = dataset_type
        self.indices = []  # Store (item_idx, frame_idx) pairs
        n_videos = 0
        
        # Filter data based on dataset_type (only for train/val, test data doesn't have dataset field)
        for item_idx, item in enumerate(data_list):
            # For test mode, skip dataset filtering (test data doesn't have 'dataset' key)
            if mode != "test":
                if dataset_type == DatasetType.EXPERT and item.get('dataset') != 'expert':
                    continue
                elif dataset_type == DatasetType.AMATEUR and item.get('dataset') != 'amateur':
                    continue
            
            n_videos += 1
            
            # For test mode, iterate through ALL frames (no labels available)
            # For train/val mode, use only labeled frames
            if mode == "test":
                num_frames = item['video'].shape[2]
                for frame_idx in range(num_frames):
                    self.indices.append((item_idx, frame_idx))
            else:
                # Use only labeled frames as centers
                labeled_frames = item.get('frames', [])
                for frame_idx in labeled_frames:
                    self.indices.append((item_idx, frame_idx))
        
        print(f"Loaded {len(self.indices)} sequences from {n_videos} videos (mode: {mode})")
        
        # Validate augmentation settings
        if self.rotation_chance > 1e-12 and mode != "train":
            raise ValueError(f"Rotation augmentation is only supported for 'train' mode, got '{mode}'")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item_idx, center_idx = self.indices[idx]
        item = self.data_list[item_idx]
        
        # Normalize the grayscale to [0, 1]
        video = item['video'].astype(np.float32) / 255.0
        num_frames = video.shape[2]
        
        # Get labels and box only for train/val modes (test data doesn't have labels)
        if self.mode != "test":
            label = item['label'].astype(np.bool_)
            box = item['box'].astype(np.bool_)
        else:
            # Test data may or may not have box, handle gracefully
            box = item.get('box', np.zeros((video.shape[0], video.shape[1]), dtype=np.bool_)).astype(np.bool_)
            label = None  # Test data has no labels

        # Choose label position in sequence
        # For test/val mode: center frame is at center of sequence (same as validation)
        # For train mode: label position can be randomized or centered
        if self.mode == "test" or self.mode == "val" or not (self.mode == "train" and self.random_label_position):
            # Test/Val mode: center frame is at center of sequence (same processing)
            half_seq = self.seq_len // 2
            start_idx = center_idx - half_seq
            sequence_label_idx = center_idx - start_idx  # This will be half_seq (center of sequence)
        else:
            # Train mode with random label position
            sequence_label_idx = random.randint(0, self.seq_len - 1)
            start_idx = center_idx - sequence_label_idx
        
        # Collect all labeled frames within sequence range
        if self.mode != "test":
            labeled_frames = item.get('frames', [])
            end_idx = start_idx + self.seq_len - 1
            all_label_indices = [sequence_label_idx]
            for labeled_frame in labeled_frames:
                if start_idx <= labeled_frame <= end_idx:
                    rel_idx = labeled_frame - start_idx
                    if rel_idx != sequence_label_idx:
                        all_label_indices.append(rel_idx)
            sequence_label_idx = sorted(all_label_indices)

        # Extract frames with reflective padding
        # Time-first layout so frames are stored as [t, h, w]
        seq = np.zeros((self.seq_len, video.shape[0], video.shape[1]), dtype=np.float32)
        if self.mode != "test":
            mask_seq = np.zeros((self.seq_len, label.shape[0], label.shape[1]), dtype=np.bool)
        else:
            mask_seq = None

        # Apply data augmentation only for training mode
        apply_rotation = False
        if self.mode == "train" and np.random.uniform(0, 1) < self.rotation_chance and self.rotation_chance > 1e-20:
            apply_rotation = True
            angle = np.random.uniform(-self.rotation_angle, self.rotation_angle)

        for t in range(self.seq_len):
            frame_t = start_idx + t
            
            # Use reflective padding (mirroring)
            if frame_t < 0:
                frame_t = -frame_t  # Mirror: -1 -> 1, -2 -> 2, etc.
            elif frame_t >= num_frames:
                frame_t = 2 * (num_frames - 1) - frame_t  # Mirror from end
            
            # Ensure frame_t is within bounds after mirroring
            frame_t = np.clip(frame_t, 0, num_frames - 1)
            
            seq[t, :, :] = video[:, :, frame_t]
            if self.mode != "test":
                mask_seq[t, :, :] = label[:, :, frame_t]

        # Apply rotation augmentation (only for train mode)
        if apply_rotation:
            seq = ndimage.rotate(seq, angle, axes=(1, 2), reshape=False, order=0, prefilter=False)
            if mask_seq is not None:
                mask_seq = ndimage.rotate(mask_seq, angle, axes=(1, 2), reshape=False, order=0, prefilter=False)
            box = ndimage.rotate(box, angle, axes=(0, 1), reshape=False, order=0, prefilter=False)
            # Safety: Re-threshold to binary (eliminate float artifacts)
            if mask_seq is not None:
                mask_seq = (mask_seq > 0.5).astype(np.bool_)
            box = (box > 0.5).astype(np.bool_)

        # Resize if target_size is specified
        if self.target_size is not None:
            seq_resized = np.zeros((self.seq_len, self.target_size[0], self.target_size[1]), dtype=np.float32)
            if mask_seq is not None:
                mask_resized = np.zeros((self.seq_len, self.target_size[0], self.target_size[1]), dtype=np.bool_)
            for t in range(self.seq_len):
                seq_resized[t, :, :] = resize_frame(seq[t, :, :], self.target_size, method='linear')
                if mask_seq is not None:
                    mask_resized[t, :, :] = resize_frame(mask_seq[t, :, :], self.target_size, method='linear')
            box_resized = resize_frame(box, self.target_size, method='nearest')
        else:
            assert seq.shape == mask_seq.shape, f"Expected seq.shape {seq.shape}, got {mask_seq.shape}"
            #all shapes always need to be divisible by 16 because of the model architecture
            assert seq.shape[1] % 16 == 0 and seq.shape[2] % 16 == 0, f"Expected seq.shape[1] and seq.shape[2] to be divisible by 16, got {seq.shape[1:3]}"
            seq_resized = seq
            mask_resized = mask_seq
            box_resized = box
        
        # Add channel dimension: (T, H, W) -> (1, T, H, W)
        assert seq_resized.shape[0] == self.seq_len, f"Expected seq_len {self.seq_len}, got {seq_resized.shape[0]}"
        seq_resized = np.expand_dims(seq_resized, axis=0)
        if mask_seq is not None:
            mask_resized = np.expand_dims(mask_resized, axis=0)
        box_resized = np.expand_dims(box_resized, axis=0)
        
        # Build result dictionary based on mode
        result = {
            'frame': torch.from_numpy(seq_resized),
            'name': item['name'],
            'orig_shape': item['video'].shape[:2],
        }
        
        # Add mode-specific fields
        if self.mode == "test":
            # Test mode: return frame, name, frame_idx, label_idx, orig_shape
            # label_idx indicates which frame in the sequence is the center frame (for mask extraction)
            result['frame_idx'] = center_idx  # Original frame index in video
            result['label_idx'] = sequence_label_idx  # Frame index in sequence (center frame)
            result['video_length'] = num_frames
        else:
            # Train/Val mode: return frame, mask, box, label_idx, video_name, orig_shape
            result['mask'] = torch.from_numpy(mask_resized)
            result['box'] = torch.from_numpy(box_resized)
            result['label_idx'] = sequence_label_idx
            result['video_name'] = item['name']
            
            # Val mode: also return original frames/mask/box for accurate IoU calculation
            if self.mode == "val":
                result['orig_frame'] = torch.from_numpy(np.expand_dims(seq, axis=0))  # (1, T, H, W)
                result['orig_mask'] = torch.from_numpy(np.expand_dims(mask_seq, axis=0))  # (1, T, H, W)
                result['orig_box'] = torch.from_numpy(np.expand_dims(box, axis=0))  # (1, H, W)
        
        return result



class DynamicAmateurMixDataset(Dataset):
    def __init__(self, expert_dataset, amateur_dataset, num_amateur_per_epoch=5, seed=None):
        """
        expert_dataset: your fixed train_ds_expert (MitralValveDataset object)
        amateur_dataset: MitralValveDataset object with all 64 amateur videos
        num_amateur_per_epoch: how many amateur videos to randomly pick each epoch
        """
        self.expert_dataset = expert_dataset
        self.amateur_dataset = amateur_dataset
        self.num_amateur_per_epoch = num_amateur_per_epoch
        self.rng = random.Random(seed)  # for reproducibility

        # Prepare indices for all amateur videos in the dataset
        self.amateur_video_indices = list(range(len(amateur_dataset)))
        self._refresh_amateur_subset()
    
    def _refresh_amateur_subset(self):
        # Randomly pick N indices for the amateur dataset
        selected_indices = self.rng.sample(self.amateur_video_indices, self.num_amateur_per_epoch)
        
        # Create a subset of the amateur dataset
        from torch.utils.data import Subset
        self.current_amateur_dataset = Subset(self.amateur_dataset, selected_indices)

        # Combine with fixed expert
        self.combined_dataset = ConcatDataset([self.expert_dataset, self.current_amateur_dataset])
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def on_epoch_end(self):
        """Call this at the end of each epoch to reshuffle amateur videos"""
        self._refresh_amateur_subset()