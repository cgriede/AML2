import os
import numpy as np
import imageio.v2 as imageio


class VideoGenerator:
    """Create GIF overlays for a batch sample.

    Expected call pattern:
        vgen = VideoGenerator(batch, pred_np, true_np)
        vgen.save_sequence_gif()
    """

    def __init__(self, batch, pred_np, true_np, batch_idx: int = 0, output_dir: str = "videos"):
        # Normalize input tensors/arrays
        self.output_dir = output_dir
        self.batch_idx = batch_idx
        os.makedirs(self.output_dir, exist_ok=True)

        # Extract frames and metadata from batch dict
        if not isinstance(batch, dict):
            raise ValueError("batch must be a dict containing 'frame' and 'video_name'")

        frames = batch.get('frame')
        if frames is None:
            raise ValueError("batch['frame'] is required")
        name_field = batch.get('video_name', batch.get('name', [f"sample_{batch_idx}"]))
        if isinstance(name_field, (list, tuple)):
            self.video_name = str(name_field[batch_idx])
        else:
            self.video_name = str(name_field)

        # frames expected shape (B, 1, T, H, W) or (B, T, H, W)
        frames_np = np.array(frames)
        sample_frames = frames_np[batch_idx]
        if sample_frames.ndim == 4 and sample_frames.shape[0] == 1:
            sample_frames = sample_frames[0]
        # sample_frames now (T, H, W) or (H, W, T)
        if sample_frames.ndim != 3:
            raise ValueError(f"Expected frames with 3 dims after squeeze, got {sample_frames.shape}")
        if sample_frames.shape[0] not in (sample_frames.shape[1], sample_frames.shape[2]):
            # assume (T, H, W)
            frames_thw = sample_frames
        else:
            # assume (H, W, T)
            frames_thw = np.transpose(sample_frames, (2, 0, 1))

        self.frames = frames_thw.astype(np.float32)
        self.T, self.H, self.W = self.frames.shape

        # Normalize masks to (T, H, W)
        self.pred_mask = self._to_thw(pred_np, batch_idx)
        self.true_mask = self._to_thw(true_np, batch_idx)

        # Align mask length with frames if needed
        self.pred_mask = self._pad_or_trim(self.pred_mask, self.T)
        self.true_mask = self._pad_or_trim(self.true_mask, self.T)

        # Use only one GT label per window if available
        label_idx_field = batch.get('label_idx')
        self.gt_center_idx = None
        if label_idx_field is not None:
            # label_idx may be tensor/list/int; normalize
            try:
                if hasattr(label_idx_field, 'cpu'):
                    idx_val = int(np.array(label_idx_field.cpu())[batch_idx].item())
                elif isinstance(label_idx_field, (list, tuple)):
                    idx_val = int(label_idx_field[0])
                else:
                    idx_val = int(label_idx_field)
                # Clamp
                self.gt_center_idx = max(0, min(self.T - 1, idx_val))
            except Exception:
                self.gt_center_idx = None

    def _to_thw(self, arr, batch_idx):
        arr_np = np.array(arr)
        if arr_np.ndim >= 5:
            arr_np = arr_np[batch_idx]
        elif arr_np.ndim >= 4 and arr_np.shape[0] != self.T:
            arr_np = arr_np[batch_idx]

        # Remove channel if present
        if arr_np.ndim == 4 and arr_np.shape[0] == 1:
            arr_np = arr_np[0]

        # Handle shapes
        if arr_np.ndim == 3:
            # possible shapes: (T, H, W) or (H, W, T)
            if arr_np.shape[0] == self.T:
                return arr_np
            if arr_np.shape[-1] == self.T:
                return np.transpose(arr_np, (2, 0, 1))
            # Single-frame broadcast (1, H, W)
            if arr_np.shape[0] == 1:
                return np.repeat(arr_np, self.T, axis=0)
        if arr_np.ndim == 2:
            # (H, W) -> broadcast across time
            return np.repeat(arr_np[None, ...], self.T, axis=0)

        raise ValueError(f"Mask array has unsupported shape {arr_np.shape}; expected (T,H,W) or (H,W,T)")

    def _pad_or_trim(self, arr, target_T):
        if arr.shape[0] == target_T:
            return arr
        if arr.shape[0] > target_T:
            return arr[:target_T]
        # pad last frame
        pad = np.repeat(arr[-1][None, ...], target_T - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)

    def _compose_frame_pair(self, t, alpha: float = 0.5):
        base = self.frames[t]
        if base.max() <= 1.5:
            base = base * 255.0
        base = np.clip(base, 0, 255).astype(np.uint8)

        # Left: ground truth overlay (blue)
        left = np.stack([base, base, base], axis=-1).astype(np.float32)
        if self.gt_center_idx is not None and t != self.gt_center_idx:
            true = np.zeros((self.H, self.W), dtype=bool)
        else:
            true = (self.true_mask[t] > 0.5)
        left[..., 2] = np.where(true, 255 * alpha + left[..., 2] * (1 - alpha), left[..., 2])
        left[..., 0] = np.where(true, left[..., 0] * (1 - alpha), left[..., 0])
        left[..., 1] = np.where(true, left[..., 1] * (1 - alpha), left[..., 1])

        # Right: prediction overlay (red)
        right = np.stack([base, base, base], axis=-1).astype(np.float32)
        pred = (self.pred_mask[t] > 0.5)
        right[..., 0] = np.where(pred, 255 * alpha + right[..., 0] * (1 - alpha), right[..., 0])
        right[..., 1] = np.where(pred, right[..., 1] * (1 - alpha), right[..., 1])
        right[..., 2] = np.where(pred, right[..., 2] * (1 - alpha), right[..., 2])

        stacked = np.concatenate([left, right], axis=1)
        return np.clip(stacked, 0, 255).astype(np.uint8)

    def save_sequence_gif(self, fps: int = 10, alpha: float = 0.5, frame_skip: int = 1) -> str:
        frame_indices = list(range(0, self.T, max(1, frame_skip)))
        frames_rgb = [self._compose_frame_pair(t, alpha) for t in frame_indices]

        filename = f"{self.video_name}_sequence_{self.batch_idx}.gif"
        output_path = os.path.join(self.output_dir, filename)
        imageio.mimsave(output_path, frames_rgb, fps=fps)
        return output_path

    # Backward-compatible alias
    def save_sequence_mp4(self, fps: int = 10, alpha: float = 0.5, frame_skip: int = 1) -> str:
        return self.save_sequence_gif(fps=fps, alpha=alpha, frame_skip=frame_skip)