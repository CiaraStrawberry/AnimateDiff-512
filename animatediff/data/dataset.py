import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
#from decord import VideoReader
import cv2

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print



class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")
            
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    

    def get_batch_simple512(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_capture = cv2.VideoCapture(video_dir)
        
        if not video_capture.isOpened():
            raise ValueError(f"Could not open video file: {video_dir}")
        
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
        
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        pixel_values = np.empty((len(batch_index), frame_height, frame_width, 3), dtype=np.uint8)
        
        for i, idx in enumerate(batch_index):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release()
                raise ValueError(f"Could not read frame {idx} from video file: {video_dir}")
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixel_values[i] = frame_rgb

        video_capture.release()
        pixel_values = torch.tensor(pixel_values).permute(0, 3, 1, 2).contiguous() * (1 / 255.0)

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name


    def get_video_info(self, idx):
        with open(self.csv_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for i, row in enumerate(csvreader):
                if i == idx + 1:  # Assuming the CSV has a header row
                    return {
                        'videoid': row[0],
                        'name': row[1],
                        'page_dir': row[3]
                    }

    def get_batch(self, idx):
        video_info = self.get_video_info(idx)
        videoid, name, page_dir = video_info['videoid'], video_info['name'], video_info['page_dir']

        video_dir = os.path.join(self.video_folder, page_dir, f"{videoid}.mp4")
        video_capture = cv2.VideoCapture(video_dir)

        if not video_capture.isOpened():
            raise ValueError(f"Could not open video file: {video_dir}")

        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        pixel_values = np.empty((len(batch_index), frame_height, frame_width, 3), dtype=np.uint8)

        for i, idx in enumerate(batch_index):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release()
                raise ValueError(f"Could not read frame {idx} from video file: {video_dir}")
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pixel_values[i] = frame_rgb

        video_capture.release()
        pixel_values = torch.tensor(pixel_values).permute(0, 3, 1, 2).contiguous() * (1 / 255.0)

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, name

    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample



if __name__ == "__main__":
    from animatediff.utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/data/webvid/results_2M_train.csv",
        video_folder="/data/webvid/data/videos",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
