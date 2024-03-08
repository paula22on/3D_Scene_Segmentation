import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.getcwd(),'../'))

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
    visualize_sample,
    visualize_100_concatenated_samples,
    write_sample_to_csv
)

SEGMENTATION = True
WEIGHTED_LOSS = False
NUM_POINTS = 4096
NUM_CLASSES = 9
NUM_TEST_TILES = 11

def main(): 
    #train_dataset = MyDataset("data/train", NUM_POINTS, "train")
    test_dataset = MyDataset("../data", NUM_POINTS, "test", sample_method = "normal")

        
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

    model_checkpoint = 'checkpoints-segmentation/segmentation_checkpoint_rotated.pth'
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

    if not os.path.exists('figures/sample_images'):
        os.makedirs('figures/sample_images')
    if not os.path.exists('figures/sample_data'):
        os.makedirs('figures/sample_data')

    with torch.no_grad():
        idx = 0
        for points, labels in test_dataloader:
            points, labels = points.to(device), labels.to(device)
            
            pred, _ = model(points)

            max_pred = torch.argmax(pred, dim=1)

            for batch_idx in range(points.size(0)):
                points_batch = points[batch_idx]
                pred_batch = max_pred[batch_idx]

                # Visualize and save original sample

                save_to_file = f'figures/sample_images/original_sample_{NUM_POINTS}points_{idx}.png'
                visualize_sample(points_batch, labels[batch_idx], save_to_file, "Testing")

                save_to_file = f'figures/sample_data/original_sample_{NUM_POINTS}points_{idx}.csv'
                labels_reshaped = labels[batch_idx].view(-1, 1)
                orig_sample = torch.cat((points_batch, labels_reshaped), dim=1)
                write_sample_to_csv(save_to_file, orig_sample)

                # Visualize and save predicted sample

                save_to_file = f'figures/sample_images/predicted_sample_{NUM_POINTS}points_{idx}.png'
                visualize_sample(points_batch, pred_batch, save_to_file, "Testing")

                save_to_file = f'figures/sample_data/predicted_sample_{NUM_POINTS}points_{idx}.csv'
                pred_reshaped = pred_batch.view(-1, 1)
                pred_sample = torch.cat((points_batch, pred_reshaped), dim=1)
                write_sample_to_csv(save_to_file, pred_sample)

                idx += 1


    # Visualize and save concatenation of 100 samples to recreate original and predicted tile
                
    for idx in range(0, NUM_TEST_TILES*100, 100):

        print(idx)

        sample_data_path  = os.path.join(os.getcwd(), 'figures/sample_data')
        sample_image_path = os.path.join(os.getcwd(), 'figures/sample_images')

        visualize_100_concatenated_samples(
            sample_data_path  = sample_data_path, 
            sample_image_path = sample_image_path, 
            filename_no_idx = f'original_sample_{NUM_POINTS}points',
            start_idx = idx
        )

        visualize_100_concatenated_samples(
            sample_data_path  = sample_data_path, 
            sample_image_path = sample_image_path, 
            filename_no_idx   = f'predicted_sample_{NUM_POINTS}points',
            start_idx = idx
        )

if __name__ == "__main__":
    main()