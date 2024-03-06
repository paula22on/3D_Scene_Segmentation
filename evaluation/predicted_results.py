import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyDataset
from model import ClassificationPointNet, SegmentationPointNet
from utils import (
    calculate_iou,
    compute_accuracy,
    compute_confusion_matrix,
    plot_accuracies,
    plot_confusion_matrix,
    plot_IoU,
    plot_iou_per_class,
    plot_losses,
    visualize_sample
)

SEGMENTATION = True
WEIGHTED_LOSS = False
NUM_POINTS = 4096
NUM_CLASSES = 9

def main(): 
    #train_dataset = MyDataset("data/train", NUM_POINTS, "train")
    test_dataset = MyDataset("data/test", NUM_POINTS, "test", sample_method = "normal")

        
    #total_length = len(train_dataset)
    #train_length = int(total_length * 0.75)
    #val_length = total_length - train_length
    #train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_length, val_length])


    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    #val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


    if SEGMENTATION:
        model = SegmentationPointNet(num_classes=NUM_CLASSES, point_dimension=3)
    else:
        model = ClassificationPointNet(
            num_classes=16, point_dimension=3, segmentation=False
        )

    if torch.cuda.is_available():
        model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_checkpoint = 'checkpoints/segmentation_checkpoint_augmentation.pth'
    if model_checkpoint:
        state = torch.load(model_checkpoint, map_location=torch.device(device))
        model.load_state_dict(state['model'])

    # Testing our model
    model.eval()
    epoch_test_loss = []
    epoch_test_acc = []
    epoch_test_iou = []
    total_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    test_iou_per_class = {
        i: [] for i in range(NUM_CLASSES)
    }  # For storing validation/testing IoU per class

    output_folder = "predicted"
    input_folder = "original"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    with torch.no_grad():
        idx = 0
        for points, labels in test_dataloader:
            points, labels = points.to(device), labels.to(device)
            
            pred, _ = model(points)

            max_pred = torch.argmax(pred, dim=1)

            for batch_idx in range(points.size(0)):
                points_batch = points[batch_idx]
                pred_batch = max_pred[batch_idx]
                save_to_file = os.path.join(input_folder,f"input_batch{batch_idx}_idx{idx}_points{NUM_POINTS}.png")
                visualize_sample(points_batch, labels[batch_idx], save_to_file, "Testing")
                save_to_file = os.path.join(output_folder,f"predicted_output_batch{batch_idx}_idx{idx}_points{NUM_POINTS}.png")
                visualize_sample(points_batch, pred_batch, save_to_file, "Testing")
                idx += 1

if __name__ == "__main__":
    main()