import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self, samples_path, number_points, split="train", sample_method="random"
    ):
        super().__init__()
        self.samples_path = samples_path + "/" + split
        self.number_points = number_points
        self.split = split
        self.sample_method = sample_method

    def __len__(self):
        # Count the number of samples (csv files) present in the directory
        num_csv_files = sum(
            1 for file in os.listdir(self.samples_path) if file.endswith(".csv")
        )
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
        sample = torch.tensor(
            sample.values, dtype=torch.float
        )  # tensor of 3 x n_points
        labels = torch.tensor(labels.values, dtype=torch.long)  # tensor of n_points

        return sample, labels

    def calculate_class_weights(self, squared = False):
            label_distribution = Counter()

            for file in os.listdir(self.samples_path):
                if file.endswith(".csv"):
                    sample_path = os.path.join(self.samples_path, file)
                    sample_df = pd.read_csv(sample_path, dtype=int)
                    labels = sample_df.iloc[:, 3].values
                    label_distribution.update(labels)

            # Calculate weights inversely proportional to the frequency of each class
            total_count = sum(
                label_distribution.values()
            )  # calculates total amount of value sin each class

            if squared:
                class_weights = {
                    class_label: total_count / (count**2)
                    for class_label, count in label_distribution.items()
                }  # total_count = total number samples, count: instances in each class
            else:
                class_weights = {
                    class_label: total_count / (len(label_distribution) * count)
                    for class_label, count in label_distribution.items()
                }  # total_count = total number samples, len(label_distr): number unique classes, count: instances in each class

            # Convert the weights into a sorted list based on the class labels -- we need to make sure that each weight corresponds to each class!!
            weights = [class_weights[i] for i in range(len(class_weights))]

            return np.array(weights)