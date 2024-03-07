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
)

SEGMENTATION = True
WEIGHTED_LOSS = False
NUM_POINTS = 2048
NUM_CLASSES = 9

def main(): 
    train_dataset = MyDataset("data", NUM_POINTS, "train")
    test_dataset = MyDataset("data", NUM_POINTS, "test")

        # Calculate weighted loss -- New code it may break here
    if WEIGHTED_LOSS:
        class_weights = (
            train_dataset.calculate_class_weights()
        )  # get weight from MyDataset function
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float
        )  # convert numpy to tensor
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = torch.nn.NLLLoss()
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

    with torch.no_grad():
        for points, labels in test_dataloader:
            points, labels = points.to(device), labels.to(device)
            pred, _ = model(points)
            loss = criterion(pred, labels)
            epoch_test_loss.append(loss.item())

            acc = compute_accuracy(pred, labels)
            epoch_test_acc.append(acc)

            ious, mean_iou = calculate_iou(pred, labels, NUM_CLASSES)
            epoch_test_iou.append(mean_iou)
            for cls in range(NUM_CLASSES):
                test_iou_per_class[cls].append(ious[cls])

            # CONFUSION MATRIX
            batch_confusion_matrix = compute_confusion_matrix(
                total_confusion_matrix, pred, labels, NUM_CLASSES
            )
            total_confusion_matrix += batch_confusion_matrix


    # After completing the loop over the test_dataloader
    average_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
    average_test_accuracy = sum(epoch_test_acc) / len(epoch_test_acc)
    average_test_iou = sum(epoch_test_iou) / len(epoch_test_iou)

    # Print the results
    print(
        f"Test Results - Loss: {average_test_loss:.4f}, Accuracy: {average_test_accuracy:.2f}, IoU: {average_test_iou:.2f}%"
    )

    # Plotting the results
    output_folder = "figures"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    class_names = ["Others", "Ground", "Vegetation", "Cars",
                   "Trucks", "Powerlines","Fences", "Poles",
                   "Buildings"]

    # testing
    plot_iou_per_class(
        test_iou_per_class,
        class_names,
        phase="Testing",
        save_to_file=os.path.join(
            output_folder, "iou_test_per_class_plot" + str(NUM_POINTS) + ".png"
        ),
    )

    plot_confusion_matrix(
        total_confusion_matrix,
        class_names,
        save_to_file=os.path.join(
            output_folder, "confmatrix_plot" + str(NUM_POINTS) + ".png"
        )
    )

if __name__ == "__main__":
    main()