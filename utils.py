import csv
import math

import laspy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from collections import Counter


# --- METRICS

def compute_accuracy(pred, target):
    """Computes accuracy of the segmentation"""
    pred_choice = pred.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    return correct.item() / float(target.size(0) * target.size(1))


def calculate_iou(pred, target, num_classes):
    """
    Calculate IoU for each class and average IoU.

    Args:
        pred: Predicted labels as a torch.Tensor.
        target: Ground truth labels as a torch.Tensor.
        num_classes: Number of classes in the dataset.

    Returns:
        ious: A list of IoU for each class.
        mean_iou: Average IoU across all classes.
    """
    ious = []
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    # Ignore background class if necessary
    for cls in range(num_classes):  # Adjust range if you have a background class
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (
            (pred_inds[target_inds]).long().sum().item()
        )  # Intersection points
        union = (
            pred_inds.long().sum().item()
            + target_inds.long().sum().item()
            - intersection
        )  # Union points

        if union == 0:
            ious.append(float("nan"))  # No division by zero
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    # Remove NaNs and calculate mean IoU
    valid_ious = [iou for iou in ious if not math.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious)

    return ious, mean_iou


def compute_confusion_matrix(batch_confusion_matrix, pred, labels, NUM_CLASSES):
    _, predicted_classes = torch.max(
        pred, 1
    )  # Get the class with the highest probability for each point/pixel
    # Ensure labels and predictions are CPU tensors and convert them to numpy
    labels = labels.cpu().numpy().flatten()  # Adjust flattening for segmentation tasks
    predicted_classes = (
        predicted_classes.cpu().numpy().flatten()
    )  # Adjust flattening for segmentation tasks
    batch_confusion_matrix = confusion_matrix(
        labels, predicted_classes, labels=np.arange(NUM_CLASSES)
    )

    return batch_confusion_matrix


# --- PLOTS


def plot_losses(train_loss, test_loss, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label="Training loss")
    plt.plot(range(epochs), test_loss, label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file, dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, label="Training accuracy")
    plt.plot(range(epochs), test_acc, label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_IoU(train_IoU, test_IoU, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_IoU)
    plt.plot(range(epochs), train_IoU, label="Training IoU")
    plt.plot(range(epochs), test_IoU, label="Validation IoU")
    plt.title("Training and validation IoU")
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_iou_per_class(iou_per_class, class_names, phase="Testing", save_to_file=None):
    """Plot IoU for each class across epochs/test."""
    epochs = range(1, len(next(iter(iou_per_class.values()))) + 1)
    plt.figure(figsize=(10, 7))
    for cls, ious in iou_per_class.items():
        plt.plot(epochs, ious, label=f"{class_names[cls]}")

    if phase=="Testing": plt.title(f"{phase} IoU per Class Over Evaluation")
    else: plt.title(f"{phase} IoU per Class Over Epochs")
    if phase=="Testing":plt.xlabel("Batch")
    else: plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)

    if save_to_file:
        plt.savefig(save_to_file)
    else: plt.show()


def plot_confusion_matrix(total_confusion_matrix, class_names, save_to_file=None):
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(
        np.round((total_confusion_matrix / total_confusion_matrix.sum(axis=1, keepdims=True))*100, 2),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Normalized Confusion Matrix (%)")
    if save_to_file:
        fig.savefig(save_to_file)
    plt.close(fig)


# --- FILE HANDLING


def convert_las_to_csv(path):
    las = laspy.read(path)
    path = path[:-3] + "csv"
    with open(path, "w") as csv_file:
        for i in range(len(las)):
            line = f"{las.X[i]},{las.Y[i]},{las.Z[i]},{las.classification[i]}\n"
            csv_file.write(line)


def write_sample_to_csv(path, sample):
    with open(path, "w") as csv_file:
        for item in sample:
            X, Y, Z, label = item
            csv_file.write(f"{X},{Y},{Z},{label}\n")


def read_sample_from_csv(path):
    points = []
    labels = []

    with open(path, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            points.append(row[:-1])
            labels.append(row[-1])

    return points, labels


# --- DATA PROCESSING


def balance_classes(sample):
    labels = sample[:, -1]
    label_distribution = Counter(labels)
    average_num_points = int(np.mean(list(label_distribution.values())))
    balanced_sample = []
    
    for label, count in label_distribution.items():
        label_samples = sample[labels == label]
        if count > average_num_points:
            indices = np.random.choice(len(label_samples), size=average_num_points, replace=False)
        else:
            indices = np.random.choice(len(label_samples), size=average_num_points-count, replace=True)
            balanced_sample.extend(label_samples) # Keep points that were already present
        balanced_sample.extend(label_samples[indices])

    return np.array(balanced_sample)


def sample_random_rotation_z_axis(points, theta=None):
    if theta is None:
        theta = np.random.uniform(0, 360)

    cos_val = np.cos(np.radians(theta))
    sin_val = np.sin(np.radians(theta))

    rotation_matrix = np.array([[cos_val, -sin_val, 0],
                                [sin_val, cos_val, 0],
                                [0, 0, 1]])  # Rotation matrix for Z-axis
    
    rotated_points = np.dot(points[:, :3], rotation_matrix)  # Apply rotation

    return np.hstack((rotated_points, points[:, 3:]))  # Reattach labels


def batch_random_rotation_z_axis(points):
    """
    Rotate points around the z-axis by a given angle.
    points: tensor of shape (batch_size, num_points, 3) representing XYZ coordinates.
    angle_degrees: rotation angle in degrees.
    """
    angle_degrees = np.random.uniform(0, 360)
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees, device=points.device))
    cos_val = torch.cos(angle_radians)
    sin_val = torch.sin(angle_radians)

    # Rotation matrix for Z-axis rotation
    rotation_matrix = torch.tensor([[cos_val, -sin_val, 0],
                                    [sin_val, cos_val, 0],
                                    [0, 0, 1]], device=points.device)
    
    # Apply rotation to each point set in the batch
    points_rotated = torch.matmul(points, rotation_matrix)

    return points_rotated


# --- VISUALIZATION


def prepare_3d_subplot(ax, points, labels, verbose=True):

    X, Y, Z, L = [], [], [], []
    for point in points:
        X.append(point[0])
        Y.append(point[1])
        Z.append(point[2])
    for label in labels:
        L.append(label)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    L = np.array(L)
    cdict = {
        1: "blue",   # Ground
        2: "green",  # Vegetation
        3: "purple", # Cars
        4: "orange", # Trucks
        5: "yellow", # Powerlines
        6: "white",  # Fences
        7: "pink",   # Poles
        8: "red",    # Buildings
    }
    for classification in np.unique(L)[1:]:
        color = cdict.get(classification, "black")
        ax.scatter(
            X[L == classification],
            Y[L == classification],
            Z[L == classification],
            s=25,
            c=color,
        )

    if verbose:
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # ax.view_init(90, 0)


def visualize_sample(points, labels, save_to_file = None, phase = "None"):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    prepare_3d_subplot(ax, points, labels)

    if phase == "Testing":
        plt.title("Predicted output")
    plt.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()


def visualize_tile_by_path(path):
    convert_las_to_csv(path)
    points, labels = read_sample_from_csv(path[:-3] + "csv")
    visualize_sample(points, labels)


def visualize_sample_by_path(path):
    points, labels = read_sample_from_csv(path)
    visualize_sample(points, labels)


def visualize_100_samples(dirpath, start_idx):
    samples = []

    fig, axes = plt.subplots(10, 10, figsize=(10, 10), subplot_kw={"projection": "3d"})

    for idx in range(start_idx, 100):
        path = f"{dirpath}/10_divisions_{idx}.csv"
        sample = read_sample_from_csv(path)
        samples.append(sample)

    for ax, sample in zip(axes.flat, samples):
        prepare_3d_subplot(ax, sample[0], sample[1], False)

    plt.tight_layout()
    plt.show()