import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import torch

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

def visualize_points(in_points, in_labels, path = None):
        
        if path is not None:
            points = []
            labels = []
            with open(path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    points.append(row[:-1])
                    labels.append(row[-1])
        else:
            points = in_points
            labels = in_labels

        X1, Y1, Z1, L1 = [], [], [], []

        for point in points:
            X1.append(point[0])
            Y1.append(point[1])
            Z1.append(point[2])

        for label in labels:
            L1.append(label)

        X1 = np.array(X1, dtype=np.uint64)
        Y1 = np.array(Y1, dtype=np.uint64)
        Z1 = np.array(Z1, dtype=np.uint64)
        L1 = np.array(L1, dtype=np.uint8)

        print(f"Number of points {len(X1)}")

        cdict = {1: 'blue', 2: 'green', 3: 'purple', 4:'orange', 5: 'yellow', 6:'white', 7:'pink', 8:'red'}  
        fig = plt.figure(figsize=[20,20])
        ax = fig.add_subplot(111, projection='3d')

        for classification in np.unique(L1)[1:]:
            color = cdict.get(classification, 'black')  
            ax.scatter(
                 X1[L1 == classification],  
                 Y1[L1 == classification],  
                 Z1[L1 == classification],  
                 s = 25, c=color)
            
        # ax.view_init(90, 0)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('LAS Point Cloud Visualization')
        plt.show()  