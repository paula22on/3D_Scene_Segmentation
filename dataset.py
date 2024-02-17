import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset, number_points, split = "train"):
        super().__init__()
        self.dataset = dataset
        self.number_points = number_points
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample_df = self.dataset[idx]

        # Randomly pick n points
        sample_reduced = sample_df.sample(self.number_points)

        # Separate point coordinates x, y, z from labels
        sample = sample_reduced.iloc[:, :3]
        labels = sample_reduced.iloc[:, 3]

        # Create tensor for the sample and its labels
        sample = torch.tensor(sample.values)  # tensor of 3 x n_points
        labels = torch.tensor(labels.values)  # tensor of n_points

        return sample, labels
