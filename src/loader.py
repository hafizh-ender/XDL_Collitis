import torch
from src.dataset import Dataset

class Loader(torch.utils.data.DataLoader):
    def __init__(self, dataframe, 
                 batch_size, 
                 num_workers, 
                 transform,
                 shuffle: bool = True,
                 seed: int = 42):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.dataloader = self.load_data()

    def load_data(self):
        dataset = Dataset(dataframe=self.dataframe, transform=self.transform, seed=self.seed, shuffle=self.shuffle)
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataframe)
