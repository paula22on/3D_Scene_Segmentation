import pandas as pd
import torch
import os
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, samples_path, number_points, split = "train"):
        super().__init__()
        self.samples_path = samples_path + "/" + split
        self.number_points = number_points
        self.split = split

    def __len__(self):
        # Count the number of samples (csv files) present in the directory
        num_csv_files = sum(1 for file in os.listdir(self.samples_path) if file.endswith('.csv'))
        return num_csv_files
    
    def __getitem__(self, idx):
        
        if idx >= self.__len__():
            raise IndexError("Trying to access sample beyond dataset length")

        # Read csv depending on index
        sample_path = f"{self.samples_path}/10_divisions_{idx}.csv"
        sample_df = pd.read_csv(sample_path, dtype=int)

        # Randomly pick n points
        sample_reduced = sample_df.sample(self.number_points)

        # Separate point coordinates x, y, z from labels
        sample = sample_reduced.iloc[:, :3]
        labels = sample_reduced.iloc[:, 3]

        # Create tensor for the sample and its labels
        sample = torch.tensor(sample.values, dtype=torch.float) # tensor of 3 x n_points
        labels = torch.tensor(labels.values, dtype=torch.long) # tensor of n_points

        return sample, labels
    