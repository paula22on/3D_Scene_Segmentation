import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, samples_path, number_points, split = "train", sample_method = "random"):
        super().__init__()
        self.samples_path = samples_path + "/" + split
        self.number_points = number_points
        self.split = split
        self.sample_method = sample_method

    def __len__(self):
        # Count the number of samples (csv files) present in the directory
        num_csv_files = sum(1 for file in os.listdir(self.samples_path) if file.endswith('.csv'))
        return num_csv_files
    
    def __getitem__(self, idx):
        
        if idx >= self.__len__():
            raise IndexError("Trying to access sample beyond dataset length")

        # Read csv depending on index
        sample_path = f"{self.samples_path}/100_divisions_{idx}.csv"
        sample_df = pd.read_csv(sample_path, dtype=int)

        # Pick n points
        if self.sample_method == "augmentation":
            sample_reduced = self.augmentation(sample_df)
        else:
            sample_reduced = sample_df.sample(self.number_points)

        # Separate point coordinates x, y, z from labels
        sample = sample_reduced.iloc[:, :3]
        labels = sample_reduced.iloc[:, 3]

        # Create tensor for the sample and its labels
        sample = torch.tensor(sample.values, dtype=torch.float) # tensor of 3 x n_points
        labels = torch.tensor(labels.values, dtype=torch.long) # tensor of n_points

        return sample, labels
    
    def augmentation(self, dataframe):

        sample = dataframe.values
        labels = sample[:, -1]
        label_distribution = Counter(labels)
        average_num_points = int(np.mean(list(label_distribution.values())))

        selected_points = []

        # Iterate over the label distribution and select samples accordingly
        for label, count in label_distribution.items():

            # If the count is above the average A, randomly select A samples
            if count > average_num_points:
                label_rows = sample[labels == label]
                selected_indices = np.random.choice(len(label_rows), size=average_num_points, replace=False)
                selected_points.extend(label_rows[selected_indices])

            # If the count is below the average A, randomly select a sample and replicate it
            elif count < average_num_points:
                label_rows = sample[labels == label]
                selected_indices = np.random.choice(len(label_rows), size=average_num_points - count, replace=True)
                selected_points.extend(label_rows)
                selected_points.extend(label_rows[selected_indices])

        sample = pd.DataFrame(selected_points)
        sample_reduced = sample.sample(self.number_points)

        return sample_reduced


    