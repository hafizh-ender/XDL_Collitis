import math
import pandas as pd
import torch
from torchvision.io import decode_image
from PIL import Image

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, is_sampling: bool = False, sampling_number: int = 50):
        if is_sampling:
            self.dataframe = dataframe.reset_index(drop=True)[:sampling_number]
        else:
            self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, len(self.dataframe)
        else:
            per_worker = int(math.ceil(len(self.dataframe) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataframe))

        for idx in range(start, end):
            row = self.dataframe.iloc[idx]
            img = decode_image(row["image_path"], mode="RGB")
            if self.transform:
                img = self.transform(img)
            label = row["class"]
            yield img, label

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        return self.dataframe.iloc[index]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 categories: list[str],
                 transform=None, 
                 seed: int = 42, 
                 shuffle: bool = True,
                 is_sampling: bool = False,
                 sampling_number: int = 50):
        self.transform = transform
        self.seed = seed
        self.is_sampling = is_sampling
        self.sampling_number = sampling_number
        self.categories = categories
        self.num_classes = len(self.categories)
        if is_sampling:
            self.dataframe = dataframe.reset_index(drop=True)[:sampling_number]
        elif shuffle:
            self.dataframe = dataframe.reset_index(drop=True).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        else:
            self.dataframe = dataframe.reset_index(drop=True)
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.zeros(self.num_classes)
        label[int(row["class"]) - 1] = 1
        assert torch.any(label), "Label tensor is empty"
        return img, label
