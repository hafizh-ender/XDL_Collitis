import math
import pandas as pd
import torch
from PIL import Image
import math

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, len(self.df)
        else:
            per_worker = int(math.ceil(len(self.df) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.df))

        for idx in range(start, end):
            row = self.df.iloc[idx]
            img = Image.open(row["image_path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = row["class"]
            yield img, label

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        return self.dataframe.iloc[index]
