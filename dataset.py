import torch
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_locs, loc_x, loc_y, radiuses, BatchSize, transform):
        super().__init__()
        self.BatchSize = BatchSize
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.radiuses = radiuses
        self.image_locs = image_locs
        self.transform = transform

    def num_of_batches(self):
        """
        Detect the total number of batches
        """
        return math.floor(len(self.list_IDs) / self.BatchSize)

    def __getitem__(self, idx):
        loc_x = self.loc_x[idx] / 100
        loc_y = self.loc_y[idx] / 100
        radius = self.radiuses[idx] / 49
        img = np.load(self.image_locs[idx] + ".npy", allow_pickle=True)
        img = np.tile(np.expand_dims(img, -1), (1, 1, 3))
        img = self.transform(img)
        return img, torch.tensor([float(loc_x), float(loc_y), float(radius)])

    def __len__(self):
        return len(self.image_locs)

