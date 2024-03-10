import csv
import math
from collections import Counter

import laspy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


# --- METRICS
def compute_accuracy(pred, target):
    """
    Computes the accuracy of segmentation predictions.

    Parameters:
        pred (torch.Tensor): The predicted labels for each pixel or point, assuming the shape [batch_size, n_classes, height, width].
        target (torch.Tensor): The ground truth labels with shape [batch_size, height, width].

    Returns:
        float: The computed accuracy.
    """
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
    """
    Computes the confusion matrix for a batch of predictions.

    Parameters:
        batch_confusion_matrix (np.array): Accumulator for the confusion matrix of the current batch.
        pred (torch.Tensor): Predicted labels for the batch.
        labels (torch.Tensor): True labels for the batch.
        NUM_CLASSES (int): Number of classes in the dataset.

    Returns:
        np.array: Updated confusion matrix after adding the current batch's results.
    """
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
    """
    Plots the training and validation loss over epochs.

    Parameters:
        train_loss (list): List of training loss values per epoch.
        test_loss (list): List of validation loss values per epoch.
        save_to_file (str, optional): Path to save the plot image. If None, the plot is shown.
    """
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label="Training loss")
    plt.plot(range(epochs), test_loss, label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file, dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    """
    Plots the training and validation accuracy over epochs.

    Parameters:
        train_acc (list): List of training accuracy values per epoch.
        test_acc (list): List of validation accuracy values per epoch.
        save_to_file (str, optional): Path to save the plot image. If None, the plot is shown.
    """
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, label="Training accuracy")
    plt.plot(range(epochs), test_acc, label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_IoU(train_IoU, test_IoU, save_to_file=None):
    """
    Plots the training and validation IoU over epochs.

    Parameters:
        train_IoU (list): List of training IoU values per epoch.
        test_IoU (list): List of validation IoU values per epoch.
        save_to_file (str, optional): Path to save the plot image. If None, the plot is shown.
    """
    fig = plt.figure()
    epochs = len(train_IoU)
    plt.plot(range(epochs), train_IoU, label="Training IoU")
    plt.plot(range(epochs), test_IoU, label="Validation IoU")
    plt.title("Training and validation IoU")
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_iou_per_class(iou_per_class, class_names, phase="Testing", save_to_file=None):
    """
    Plots the IoU for each class across epochs or tests.

    Parameters:
        iou_per_class (dict): Dictionary where keys are class indices and values are lists of IoU values.
        class_names (list): List of class names corresponding to indices.
        phase (str): Phase of the plot, "Testing" or another phase to adjust the title.
        save_to_file (str, optional): Path to save the plot image. If None, the plot is shown.
    """
    epochs = range(1, len(next(iter(iou_per_class.values()))) + 1)
    plt.figure(figsize=(10, 7))
    for cls, ious in iou_per_class.items():
        plt.plot(epochs, ious, label=f"{class_names[cls]}")

    if phase == "Testing":
        plt.title(f"{phase} IoU per Class Over Evaluation")
    else:
        plt.title(f"{phase} IoU per Class Over Epochs")
    if phase == "Testing":
        plt.xlabel("Batch")
    else:
        plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)

    if save_to_file:
        plt.savefig(save_to_file)
    else:
        plt.show()


def plot_confusion_matrix(total_confusion_matrix, class_names, save_to_file=None):
    """
    Plots a normalized confusion matrix.

    Parameters:
        total_confusion_matrix (np.array): Confusion matrix to be plotted.
        class_names (list): Names corresponding to each class index.
        save_to_file (str, optional): Path to save the plot image. If None, the plot is shown.
    """
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(
        np.round(
            (total_confusion_matrix / total_confusion_matrix.sum(axis=1, keepdims=True))
            * 100,
            2,
        ),
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
    """
    Converts a LAS file to a CSV file with columns for X, Y, Z coordinates and classification.

    Args:
        path (str): The file path of the LAS file to convert.
    """
    las = laspy.read(path)
    path = path[:-3] + "csv"
    with open(path, "w") as csv_file:
        for i in range(len(las)):
            line = f"{las.X[i]},{las.Y[i]},{las.Z[i]},{las.classification[i]}\n"
            csv_file.write(line)


def write_sample_to_csv(path, sample):
    """
    Writes a sample of point cloud data to a CSV file.

    Args:
        path (str): The file path of the CSV file to write.
        sample (list of tuples): A list where each tuple represents a point (X, Y, Z) and its label.
    """
    with open(path, "w") as csv_file:
        for item in sample:
            X, Y, Z, label = item
            csv_file.write(f"{X},{Y},{Z},{label}\n")


def read_sample_from_csv(path):
    """
    Reads point cloud data and labels from a CSV file.

    Args:
        path (str): The file path of the CSV file to read.

    Returns:
        tuple: A tuple containing two lists, one for the points and another for their labels.
    """
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
    """
    Balances classes in a dataset by under-sampling or over-sampling to the average class count.

    Args:
        sample (np.array): An array of shape (N, 4) where N is the number of points and each row is (X, Y, Z, label).

    Returns:
        np.array: A balanced array of shape (M, 4) where M <= N.
    """
    labels = sample[:, -1]
    label_distribution = Counter(labels)
    average_num_points = int(np.mean(list(label_distribution.values())))
    balanced_sample = []

    for label, count in label_distribution.items():
        label_samples = sample[labels == label]
        if count > average_num_points:
            indices = np.random.choice(
                len(label_samples), size=average_num_points, replace=False
            )
        else:
            indices = np.random.choice(
                len(label_samples), size=average_num_points - count, replace=True
            )
            balanced_sample.extend(
                label_samples
            )  # Keep points that were already present
        balanced_sample.extend(label_samples[indices])

    return np.array(balanced_sample)


def sample_random_rotation_z_axis(points, theta=None):
    """
    Rotates a sample of points around the Z-axis by a random angle or a given angle.

    Args:
        points (np.array): An array of points of shape (N, 4), where N is the number of points.
        theta (float, optional): The rotation angle in degrees. If None, a random angle is used.

    Returns:
        np.array: The rotated points of shape (N, 4).
    """
    if theta is None:
        theta = np.random.uniform(0, 360)

    cos_val = np.cos(np.radians(theta))
    sin_val = np.sin(np.radians(theta))

    rotation_matrix = np.array(
        [[cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]]
    )  # Rotation matrix for Z-axis

    rotated_points = np.dot(points[:, :3], rotation_matrix)  # Apply rotation

    return np.hstack((rotated_points, points[:, 3:]))  # Reattach labels


def batch_random_rotation_z_axis(points):
    """
    Rotates a batch of points around the z-axis by a random angle.

    Parameters:
    - points (torch.Tensor): A tensor of shape (batch_size, num_points, 3) representing
      XYZ coordinates of points for each instance in the batch.

    Returns:
    - torch.Tensor: The rotated points with the same shape as the input.
    """
    angle_degrees = np.random.uniform(0, 360)
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees, device=points.device))
    cos_val = torch.cos(angle_radians)
    sin_val = torch.sin(angle_radians)

    # Rotation matrix for Z-axis rotation
    rotation_matrix = torch.tensor(
        [[cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]], device=points.device
    )

    # Apply rotation to each point set in the batch
    points_rotated = torch.matmul(points, rotation_matrix)

    return points_rotated


# --- VISUALIZATION
def prepare_3d_subplot(ax, points, labels, verbose=True, top_view=False):
    """
    Prepares a 3D subplot with labeled point cloud data.
    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot on.
    - points (list of lists): The point cloud data, where each point is represented as [x, y, z].
    - labels (list): The labels for each point in the point cloud.
    - verbose (bool, optional): If True, axis labels are set; otherwise, axis ticks are removed.
    Returns:
    - None
    """
    X, Y, Z, L = [], [], [], []
    for point in points:
        X.append(point[0])
        Y.append(point[1])
        Z.append(point[2])
    for label in labels:
        L.append(label)
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    Z = np.array(Z).astype(float)
    L = np.array(L).astype(float)
    cdict = {
        1: "blue",  # Ground
        2: "green",  # Vegetation
        3: "purple",  # Cars
        4: "orange",  # Trucks
        5: "yellow",  # Powerlines
        6: "gray",  # Fences
        7: "pink",  # Poles
        8: "red",  # Buildings
    }
    clabel = {
        1: "Ground",  # Ground
        2: "Vegetation",  # Vegetation
        3: "Cars",  # Cars
        4: "Trucks",  # Trucks
        5: "Powerlines",  # Powerlines
        6: "Fences",  # Fences
        7: "Poles",  # Poles
        8: "Buildings",  # Buildings
    }
    for classification in np.unique(L)[1:]:
        color = cdict.get(classification, "black")
        ax.scatter(
            X[L == classification],
            Y[L == classification],
            Z[L == classification],
            s=5 if top_view else 25,
            c=color,
            label = clabel.get(classification, "black")
        )
    if verbose:
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.legend()
    if top_view:
        ax.view_init(90, 0)
        ax.set_zticks([])
    


def visualize_sample(points, labels, save_to_file = None, phase = None, top_view = False):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    prepare_3d_subplot(ax, points, labels, top_view=top_view)

    if phase == "Testing":
        plt.title("Predicted output")
    plt.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file)
        plt.close()
    else:
        plt.show()


def visualize_tile_by_path(path):
    """
    Converts a LAS file to CSV, then visualizes the point cloud data from the generated CSV file.

    Parameters:
    - path (str): Path to the LAS file to be visualized.

    Returns:
    - None
    """
    convert_las_to_csv(path)
    points, labels = read_sample_from_csv(path[:-3] + "csv")
    visualize_sample(points, labels)


def visualize_sample_by_path(path):
    """
    Visualizes the point cloud data from a given CSV file.

    Parameters:
    - path (str): Path to the CSV file containing point cloud data.

    Returns:
    - None
    """
    points, labels = read_sample_from_csv(path)
    visualize_sample(points, labels)


def visualize_100_subsamples(dirpath, start_idx):
    """
    Visualizes 100 point cloud samples from CSV files located in a directory, starting from a specified index.

    Parameters:
    - dirpath (str): Directory path containing the CSV files.
    - start_idx (int): The starting index to begin visualization.

    Returns:
    - None
    """
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


def visualize_100_concatenated_samples(sample_data_path, sample_image_path, filename_no_idx, start_idx):
    """
    Concatenates and visualizes 100 point cloud samples from CSV files located in a directory, starting from a specified index.
    Stores the resulting image into the specified directory.

    Parameters:
    - sample_data_path (str): Directory path containing the CSV files and output path for the CSV containing concatenated samples.
    - sample_image_path (str): Directory output path for the PNG resulting image.
    - filename_no_idx (str): Name of the files without index. Expected to be the same for all 100 files.
    - start_idx (int): The starting index to begin concatenation.
    
    Returns:
    - None
    """
    filelist = [f'{sample_data_path}/{filename_no_idx}_{i}.csv' for i in range(start_idx,start_idx+100)]

    outpath_csv = f'{sample_data_path}/{filename_no_idx}_{start_idx}_to_{start_idx+100-1}.csv'
    outpath_png = f'{sample_image_path}/{filename_no_idx}_{start_idx}_to_{start_idx+100-1}.png'

    with open(outpath_csv, 'w') as outfile:
        for filename in filelist:
            with open(filename, 'r') as infile:
                for line in infile:
                    outfile.write(line)

    points, labels = read_sample_from_csv(outpath_csv)
    visualize_sample(points, labels, save_to_file=outpath_png, top_view=True)