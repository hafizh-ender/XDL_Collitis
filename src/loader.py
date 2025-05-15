import torch
from src.dataset import IterableDataset
class Loader():
    def __init__(self, dataframe, 
                 batch_size, 
                 num_workers, 
                 transform,
                 shuffle: bool = True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle

    def load_data(self):
        dataset = IterableDataset(dataframe=self.dataframe, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=self.batch_size, 
                                           shuffle=self.shuffle, 
                                           num_workers=self.num_workers)
        return dataloader
