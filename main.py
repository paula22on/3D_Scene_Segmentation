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
    plot_losses,
    visualize_points,
)

SEGMENTATION = True
WEIGHTED_LOSS = True
NUM_POINTS = 2048
NUM_CLASSES = 9

train_dataset = MyDataset("data", NUM_POINTS, "train", "augmentation")
test_dataset = MyDataset("data", NUM_POINTS, "test")
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [2300, 600])

# ---- Sample visualization
# points, labels = test_dataset[0]
# points, labels = points.tolist(), labels.tolist()
# visualize_points(points, labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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

optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint_dir = "checkpoints-segmentation"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

iou_log_file = open("iou_log.txt", "a")

# ---- All above code works! Currently testing...
# Training and Evaluation Loop
epochs = 80
train_loss = []
test_loss = []
train_acc = []
test_acc = []
train_iou = []
test_iou = []
best_loss = np.inf

for epoch in tqdm(range(epochs)):
    model.train()
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_train_iou = []

    # Training Loop
    for i, data in enumerate(train_dataloader):
        points, labels = data
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred, _ = model(points)

        loss = criterion(pred, labels)
        epoch_train_loss.append(loss.item())

        # Accuracy Calculation for Segmentation
        acc = compute_accuracy(pred, labels)
        epoch_train_acc.append(acc)

        # IoU Calculation for Segmentation
        ious, mean_iou = calculate_iou(pred, labels, NUM_CLASSES)
        epoch_train_iou.append(mean_iou)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(
            f"Training - Epoch {epoch}, Batch {i}: Train Loss: {loss.item()}, Train Acc: {acc}"
        )

    epoch_val_loss = []
    epoch_val_acc = []
    epoch_val_iou = []

    # Validation Loop
    with torch.no_grad():
        for data in val_dataloader:
            points, labels = data
            points, labels = points.to(device), labels.to(device)
            pred, _ = model(points)
            loss = criterion(pred, labels)
            epoch_val_loss.append(loss.item())

            # Caclulate Accuracy and append to epoch_val_iou
            acc = compute_accuracy(pred, labels)
            epoch_val_acc.append(acc)

            # Caclulate IoU and append to epoch_val_iou
            ious, mean_iou = calculate_iou(pred, labels, NUM_CLASSES)
            epoch_val_iou.append(mean_iou)

        print(
            "Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f, train IoU %s,  val IoU: %f"
            % (
                epoch,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_val_loss), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_val_acc), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_val_iou), 4),
            )
        )

    # Calculate the average IoU for the epoch
    avg_epoch_val_iou = sum(epoch_val_iou) / len(epoch_val_iou)
    iou_log_file.write(f"Epoch {epoch + 1}: avg_iou = {avg_epoch_val_iou:.4f}\n")
    iou_log_file.flush()

    # Checkpoint Saving
    avg_test_loss = np.mean(epoch_val_loss)
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(
            state,
            os.path.join(checkpoint_dir, f"segmentation_checkpoint_epoch_{epoch}.pth"),
        )

    # Logging
    train_loss.append(np.mean(epoch_train_loss))
    test_loss.append(np.mean(epoch_val_loss))
    train_acc.append(np.mean(epoch_train_acc))
    test_acc.append(np.mean(epoch_val_acc))
    train_iou.append(np.mean(epoch_train_iou))
    test_iou.append(np.mean(epoch_val_iou))

    print(
        f"Epoch {epoch}: Train Loss: {train_loss[-1]}, "
        f"Val Loss: {test_loss[-1]}, "
        f"Train Acc: {train_acc[-1]}, "
        f"Val Acc: {test_acc[-1]}, "
        f"Train IoU: {train_iou[-1]}, "
        f"Val IoU: {test_iou[-1]}"
    )

iou_log_file.close()

# Testing our model
model.eval()
epoch_test_loss = []
epoch_test_acc = []
epoch_test_iou = []
total_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

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

plot_losses(
    train_loss,
    test_loss,
    save_to_file=os.path.join(output_folder, "loss_plot" + str(NUM_POINTS) + ".png"),
)
plot_accuracies(
    train_acc,
    test_acc,
    save_to_file=os.path.join(
        output_folder, "accuracy_plot" + str(NUM_POINTS) + ".png"
    ),
)
plot_IoU(
    train_iou,
    test_iou,
    save_to_file=os.path.join(output_folder, "iou_plot" + str(NUM_POINTS) + ".png"),
)

class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

plot_confusion_matrix(
    total_confusion_matrix,
    class_names,
    save_to_file=os.path.join(
        output_folder, "confmatrix_plot" + str(NUM_POINTS) + ".png"
    ),
)
