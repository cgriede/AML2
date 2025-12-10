"""
Data loading and preprocessing for Mitral Valve Segmentation
"""

import numpy as np
import torch
import gzip
import pickle
from torch.utils.data import Dataset
from scipy import ndimage
from enum import Enum


def load_zipped_pickle(filename):
    """Load pickled data from gzip file"""
    with gzip.open(filename, 'rb') as f:
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
    """Load temporal sequences from train.pkl - lazy loading, compute on demand"""
    
    def __init__(self, data_list, seq_len: int=11, target_size: tuple=None, dataset_type: DatasetType=DatasetType.EXPERT):
        self.data_list = data_list
        self.seq_len = seq_len
        self.target_size = target_size
        self.dataset_type = dataset_type
        self.indices = []  # Store (item_idx, frame_idx) pairs
        
        # Filter data based on dataset_type
        for item_idx, item in enumerate(data_list):
            # Skip items that don't match the desired dataset type
            if dataset_type == DatasetType.EXPERT and item['dataset'] != 'expert':
                continue
            elif dataset_type == DatasetType.AMATEUR and item['dataset'] != 'amateur':
                continue
            # MIXED includes both, so no filtering needed
            
            # Add all frames as potential window centers
            num_frames = item['video'].shape[2]
            for frame_idx in range(num_frames):
                self.indices.append((item_idx, frame_idx))
        
        print(f"Loaded {len(self.indices)} sequences from {dataset_type.value} dataset")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item_idx, center_idx = self.indices[idx]
        item = self.data_list[item_idx]
        
        # Normalize the grayscale to [0, 1]
        video = item['video'].astype(np.float32) / 255.0
        label = item['label'].astype(np.float32)
        box = item['box'].astype(np.float32)
        num_frames = video.shape[2]
        
        # Get seq_len consecutive frames centered on frame
        half_seq = self.seq_len // 2
        start_idx = center_idx - half_seq
        end_idx = center_idx + half_seq + 1  # +1 because slice is exclusive
        
        # Extract frames with reflective padding
        seq = np.zeros((video.shape[0], video.shape[1], self.seq_len), dtype=np.float32)
        mask_seq = np.zeros((label.shape[0], label.shape[1], self.seq_len), dtype=np.float32)
        
        for t in range(self.seq_len):
            frame_t = start_idx + t
            
            # Use reflective padding (mirroring)
            if frame_t < 0:
                frame_t = -frame_t  # Mirror: -1 -> 1, -2 -> 2, etc.
            elif frame_t >= num_frames:
                frame_t = 2 * (num_frames - 1) - frame_t  # Mirror from end
            
            # Ensure frame_t is within bounds after mirroring
            frame_t = np.clip(frame_t, 0, num_frames - 1)
            
            seq[:, :, t] = video[:, :, frame_t]
            mask_seq[:, :, t] = label[:, :, frame_t]
        
        # Resize if target_size is specified
        if self.target_size is not None:
            seq_resized = np.zeros((self.target_size[0], self.target_size[1], self.seq_len), dtype=np.float32)
            mask_resized = np.zeros((self.target_size[0], self.target_size[1], self.seq_len), dtype=np.float32)
            for t in range(self.seq_len):
                # Linear interpolation for frames and masks
                seq_resized[:, :, t] = resize_frame(seq[:, :, t], self.target_size, method='linear')
                mask_resized[:, :, t] = resize_frame(mask_seq[:, :, t], self.target_size, method='linear')
            
            # Nearest neighbor for bounding box (preserve sharp boundaries)
            box_resized = resize_frame(box, self.target_size, method='nearest')
        else:
            seq_resized = seq
            mask_resized = mask_seq
            box_resized = box
        
        # Add channel dimension: (H, W, T) -> (1, H, W, T)
        seq_resized = np.expand_dims(seq_resized, axis=0)
        mask_resized = np.expand_dims(mask_resized, axis=0)
        box_resized = np.expand_dims(box_resized, axis=0)
        
        return {
            'frame': torch.from_numpy(seq_resized).to(torch.float16),
            'mask': torch.from_numpy(mask_resized).to(torch.bool),
            'box': torch.from_numpy(box_resized).to(torch.bool),
            'label_idx': [center_idx], # only one label index for now
            'video_name': item['name']
        }


class TestDataset(Dataset):
    """Load temporal sequences from test.pkl - lazy loading, compute on demand"""
    
    def __init__(self, data_list, seq_len: int=11, target_size: tuple=None):
        self.data_list = data_list
        self.seq_len = seq_len
        self.target_size = target_size
        self.indices = []  # Store (item_idx, frame_idx) pairs
        
        for item_idx, item in enumerate(data_list):
            num_frames = item['video'].shape[2]
            for frame_idx in range(num_frames):
                self.indices.append((item_idx, frame_idx))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item_idx, center_idx = self.indices[idx]
        item = self.data_list[item_idx]
        
        # Normalize the grayscale to [0, 1]
        video = item['video'].astype(np.float32) / 255.0
        num_frames = video.shape[2]
        
        # Get seq_len consecutive frames centered on frame
        half_seq = self.seq_len // 2
        start_idx = center_idx - half_seq
        end_idx = center_idx + half_seq + 1
        
        # Extract frames with reflective padding
        seq = np.zeros((video.shape[0], video.shape[1], self.seq_len), dtype=np.float32)
        
        for t in range(self.seq_len):
            frame_t = start_idx + t
            
            # Use reflective padding (mirroring)
            if frame_t < 0:
                frame_t = -frame_t  # Mirror: -1 -> 1, -2 -> 2, etc.
            elif frame_t >= num_frames:
                frame_t = 2 * (num_frames - 1) - frame_t  # Mirror from end
            
            # Ensure frame_t is within bounds after mirroring
            frame_t = np.clip(frame_t, 0, num_frames - 1)
            
            seq[:, :, t] = video[:, :, frame_t]
        
        # Resize to target size
        if self.target_size is not None:
            seq_resized = np.zeros((self.target_size[0], self.target_size[1], self.seq_len), dtype=np.float32)
            for t in range(self.seq_len):
                seq_resized[:, :, t] = resize_frame(seq[:, :, t], self.target_size, method='linear')
        else:
            seq_resized = seq
        
        # Add channel dimension: (H, W, T) -> (1, H, W, T)
        seq_resized = np.expand_dims(seq_resized, axis=0)
        
        return {
            'frame': torch.from_numpy(seq_resized).to(torch.float16),
            'name': item['name'],
            'frame_idx': center_idx,
            'orig_shape': item['video'].shape[:2]
        }