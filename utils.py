import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import torch
import laspy


# --- METRICS

def compute_accuracy(pred, target):
    """Computes accuracy of the segmentation"""
    pred_choice = pred.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    return correct.item() / float(target.size(0) * target.size(1))


def plot_losses(train_loss, test_loss, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, label='Training loss')
    plt.plot(range(epochs), test_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file,dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, label='Training accuracy')
    plt.plot(range(epochs), test_acc, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)

def plot_IoU(train_IoU, test_IoU, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_IoU)
    plt.plot(range(epochs), train_IoU, label='Training IoU')
    plt.plot(range(epochs), test_IoU, label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)

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
        intersection = (pred_inds[target_inds]).long().sum().item()  # Intersection points
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection  # Union points
        
        if union == 0:
            ious.append(float('nan'))  # No division by zero
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    
    # Remove NaNs and calculate mean IoU
    valid_ious = [iou for iou in ious if not math.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious)
    
    return ious, mean_iou


# --- FILE HANDLING

def convert_las_to_csv(path):
    las = laspy.read(path)
    path = path[:-3]+"csv"
    with open(path, "w") as csv_file:
        for i in range(len(las)):
            line = f"{las.X[i]},{las.Y[i]},{las.Z[i]},{las.classification[i]}\n"
            csv_file.write(line)


def read_sample_from_csv(path):
    points = []
    labels = []

    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            points.append(row[:-1])
            labels.append(row[-1])

    return points, labels


# --- VISUALIZATION
        
def prepare_3d_subplot(ax, points, labels, verbose = True):
    X, Y, Z, L = [],[],[],[]
    for point in points:
        X.append(point[0])
        Y.append(point[1])
        Z.append(point[2])
    for label in labels:
        L.append(label)
    
    X = np.array(X, dtype=np.uint64)
    Y = np.array(Y, dtype=np.uint64)
    Z = np.array(Z, dtype=np.uint64)
    L = np.array(L, dtype=np.uint8)
    
    cdict = {1: 'blue', 2: 'green', 3: 'purple', 4:'orange', 5: 'yellow', 6:'white', 7:'pink', 8:'red'}  
    for classification in np.unique(L)[1:]:
        color = cdict.get(classification, 'black')  
        ax.scatter(
            X[L == classification],  
            Y[L == classification],  
            Z[L == classification],  
            s = 25,
            c=color
        )

    if verbose:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # ax.view_init(90, 0)


def visualize_sample(points, labels):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    prepare_3d_subplot(ax, points, labels)
    plt.tight_layout()
    plt.show()


def visualize_tile_by_path(path):
    convert_las_to_csv(path)
    points, labels = read_sample_from_csv(path[:-3]+"csv")
    visualize_sample(points, labels)


def visualize_sample_by_path(path):
    points, labels = read_sample_from_csv(path)
    visualize_sample(points, labels)


def visualize_100_samples(dirpath, start_idx):
    samples = []

    fig, axes = plt.subplots(
        10, 10, figsize=(10, 10), subplot_kw={'projection': '3d'}
    )

    for idx in range(start_idx, 100):
        path = f"{dirpath}/10_divisions_{idx}.csv"
        sample = read_sample_from_csv(path)
        samples.append(sample)

    for ax, sample in zip(axes.flat, samples):
        prepare_3d_subplot(ax, sample[0], sample[1], False)

    plt.tight_layout()
    plt.show()
