import matplotlib.pyplot as plt
import numpy as np
import csv

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