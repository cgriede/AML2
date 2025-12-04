import pickle
import gzip
from tkinter import NO
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML  # For rendering animation in Jupyter
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


def preprocess_train_data(data):
    """
    This function selects the frames of the video that are labeled and creates the video and mask frames.
    """
    video_frames = []
    mask_frames = []
    names = []
    boxes = []
    label_types = []
    for item in tqdm(data):
        video = item['video']
        name = item['name']
        box = item['box']
        label_type = item['dataset']

        height, width, n_frames = video.shape
        mask = np.zeros((height, width, n_frames), dtype=np.bool)
        for frame in item['frames']:
            mask[:, :, frame] = item['label'][:, :, frame]
            video_frame = video[:, :, frame]
            mask_frame = mask[:, :, frame]
            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.int32)
            video_frames.append(video_frame)
            mask_frames.append(mask_frame)
            names.append(name)
            boxes.append(box)
            label_types.append(label_type)
    return names, video_frames, mask_frames, boxes, label_types

def preprocess_test_data(data):
    video_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        video = video.astype(np.float32).transpose((2, 0, 1))
        video = np.expand_dims(video, axis=3)
        video_frames += list(video)
        names += [item['name'] for _ in video]
    return names, video_frames



def get_sequences(arr):
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

def submit():
    raise NotImplementedError("Not implemented")
    submission_file = "mysubmissionfile.csv"
    cmd = f"""
    kaggle competitions submit -c eth-aml-2025-project-task-2 -f {submission_file} -m "Message"
    """
    #create submission file
    df = pd.DataFrame({"id":ids, "value":[list(map(int, minili)) for minili in values]})
    df.to_csv(submission_file, index=False)
    os.system(cmd)


class ValveSegmentationModel:
    def __init__(self):
        #TODO:create the data attributes for the model
        self.videos_raw = None
        self.masks = None
        self.names = None

        #TODO: add / import the Neural Network model


    def train(self):
        #TODO: train the model
        pass

    def predict(self):
        #TODO: predict the masks
        pass

    def evaluate(self) -> tuple[int, list[list[tuple[int, int]]]]:
        #TODO: evaluate the model
        pass

def get_boxed_videos(videos, boxes):
    """
    Returns videos with pixels outside the box set to 0.
    
    Args:
        videos: Either a single video array with shape (height, width, frames) 
               or a list of video frames, each with shape (height, width, channels)
        boxes: Either a single boolean array with shape (height, width) 
               or a list of boolean arrays, each with shape (height, width) where True indicates inside the box
    
    Returns:
        If input is a single video: returns a single boxed video array with same shape
        If input is a list: returns a list of boxed video frames with same shape as input videos
    """
    # Handle single video array case
    if isinstance(videos, np.ndarray) and isinstance(boxes, np.ndarray):
        # video is (height, width, frames), box is (height, width)
        # Expand box to (height, width, 1) to broadcast with video
        box_expanded = np.expand_dims(boxes, axis=-1)
        return videos * box_expanded.astype(videos.dtype)
    
    # Handle list case
    boxed_videos = []
    for video, box in zip(videos, boxes):
        # Expand box to match video dimensions (height, width, channels)
        # box is (height, width), video is (height, width, frames)
        # Use the mask directly on the video frames
        boxed_video = video * box[..., None]
        boxed_videos.append(boxed_video)
    return boxed_videos

def main():
    # load data
    train_data = load_zipped_pickle("train.pkl")
    test_data = load_zipped_pickle("test.pkl")
    samples = pd.read_csv("sample.csv")
    for key, value in train_data[1].items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
            if key == "box":
                true_box_indices = np.argwhere(value)
                print(f"Shape of box (True indices): {true_box_indices.shape}")
                print(f"Total elements in box: {value.size}")
                print(f"Number of True values: {np.sum(value)}")
                print(f"Percentage True: {100 * np.sum(value) / value.size:.2f}%")
                if len(true_box_indices) > 0:
                    min_row, min_col = true_box_indices.min(axis=0)
                    max_row, max_col = true_box_indices.max(axis=0)
                    print(f"Bounding box region: rows [{min_row}, {max_row}], cols [{min_col}, {max_col}]")
                    print(f"Bounding box dimensions: {max_row - min_row + 1} x {max_col - min_col + 1}")
                    print(f"Expected True count if rectangular: {(max_row - min_row + 1) * (max_col - min_col + 1)}")
                    print(f"Actual True count: {len(true_box_indices)}")
                    print(f"Difference (holes?): {(max_row - min_row + 1) * (max_col - min_col + 1) - len(true_box_indices)}")
        else:
            print(f"{key}: {value}")

    boxed_video = get_boxed_videos(train_data[1]['video'], train_data[1]['box'])
    print(f"Shape of boxed video: {boxed_video.shape}")
    print(f"Shape of original video: {train_data[1]['video'].shape}")

    names, videos, masks, boxes, label_types = preprocess_train_data(train_data)
    test_names, test_videos = preprocess_test_data(test_data)

_train_data_structure = {
    "name": "str, video name",
    "video": "array[height, width, number of frames], video frames",
    "box": "array[height, width], bounding box of the valve",
    "label": "array[height, width, number of frames], label of the valve",
    "frames": "list[int], labeled frames indices",
    "dataset": ("amateur", "expert")
}
_test_data_structure = {
    "name": "str, video name",
    "video": "array[height, width, number of frames], video frames",
}


if __name__ == "__main__":
    main()