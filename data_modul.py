import os
import random

import numpy as np
from torch.utils.data import DataLoader, Dataset
import lightning as L


class NDataset(Dataset):
    def __init__(
            self,
            data_paths,
    ):
        super().__init__()
        self.root_dir = data_paths
        self.data_files = []

        for subdir in os.listdir(data_paths):
            subdir_path = os.path.join(data_paths, subdir)
            if os.path.isdir(subdir_path):
                model_dir = os.path.join(subdir_path, 'model')
                self.data_files.extend(
                    [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        model_path = self.data_files[idx]
        data_path = model_path.replace('model', 'data')
        height_path = model_path.replace('model', 'height')

        model = np.load(model_path)
        model = np.log10(model)
        model = model / 3

        data = np.load(data_path)
        data = np.abs(data)
        if np.any(data == 0):
            print(f"Zero values found in data path: {data_path}")
        data = np.log10(data)

        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) * 2 / (data_max - data_min) - 1

        model = model[np.newaxis, :]
        data = data[np.newaxis, :]

        model = (model.astype(np.float32))
        data = (data.astype(np.float32))

        height_data = np.load(height_path)
        height = height_data.astype(np.float32)

        return {'model': model, 'data': data, 'height': height}


class DataModule(L.LightningDataModule):
    def __init__(self, train_dir: str = "./test_data", val_dir: str = "./test_data", batch_size: int = 1,
                 num_workers: int = 0):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = NDataset(data_paths=self.train_dir)
        self.val_set = NDataset(data_paths=self.val_dir)

    def train_dataloader(self):
        ld_train = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return ld_train

    def val_dataloader(self):
        ld_val = DataLoader(
            self.val_set,
            num_workers=0,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return ld_val
