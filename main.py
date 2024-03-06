import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CHECKPOINT_DIRECTORY,
    FIGURES_DIRECTORY,
    NUM_CLASSES,
    NUM_CLASSES_CLASSIFICATION,
    NUM_POINTS,
    RANDOM_ROTATION_IN_BATCH,
    SEGMENTATION,
    WEIGHTED_LOSS,
)
from dataset import MyDataset
from model import ClassificationPointNet, SegmentationPointNet
from utils import (
    batch_random_rotation_z_axis,
    calculate_iou,
    compute_accuracy,
    compute_confusion_matrix,
    plot_accuracies,
    plot_confusion_matrix,
    plot_IoU,
    plot_iou_per_class,
    plot_losses,
)


def main():
    # Prepare datasets for training, validation, and testing.
    train_dataset = MyDataset("data", NUM_POINTS, "train")
    test_dataset = MyDataset("data", NUM_POINTS, "test")

    # Option to use weighted loss for dealing with imbalanced classes.
    if WEIGHTED_LOSS:
        class_weights = (
            train_dataset.calculate_class_weights()
        )  # Calculate class weights
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float
        )  # Convert to tensor
        if torch.cuda.is_available():
            class_weights_tensor = (
                class_weights_tensor.cuda()
            )  # Move to GPU if available

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = (
            torch.nn.NLLLoss()
        )  # Use Negative Log Likelihood Loss if weighted loss is not enabled

    # Split the training dataset into training and validation sets.
    total_length = len(train_dataset)
    train_length = int(total_length * 0.75)
    val_length = total_length - train_length
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_length, val_length]
    )

    # Prepare data loaders for training, validation, and testing phases.
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # Initialize the model based on the task (segmentation or classification).
    if SEGMENTATION:
        model = SegmentationPointNet(num_classes=NUM_CLASSES, point_dimension=3)
    else:
        model = ClassificationPointNet(
            num_classes=NUM_CLASSES_CLASSIFICATION,
            point_dimension=3,
            segmentation=False,
        )

    # Move model to GPU if available, else use CPU.
    if torch.cuda.is_available():
        model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    # Set up the optimizer for model parameters.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Ensure the checkpoint directory exists.
    checkpoint_dir = CHECKPOINT_DIRECTORY
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize variables for logging and tracking progress.
    epochs = 80
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_iou = []
    test_iou = []
    best_loss = np.inf  # Track the best loss for model saving
    train_iou_per_class = {
        i: [] for i in range(NUM_CLASSES)
    }  # IoU per class for training
    test_iou_per_class = {
        i: [] for i in range(NUM_CLASSES)
    }  # IoU per class for testing/validation
    iou_log_file = open("iou_log.txt", "a")  # File to log IoU over epochs

    # Training and validation loop
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model to training mode
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_train_iou = []
        epoch_train_iou_per_class = {i: [] for i in range(NUM_CLASSES)}

        # Training Loop
        for i, data in enumerate(train_dataloader):
            points, labels = data  # Unpack data
            points, labels = points.to(device), labels.to(
                device
            )  # Move data to the appropriate device
            if RANDOM_ROTATION_IN_BATCH:
                points = batch_random_rotation_z_axis(points)  # Data augmentation

            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            pred, _ = model(points)
            loss = criterion(pred, labels)  # Compute loss
            epoch_train_loss.append(loss.item())

            # Compute accuracy and IoU for segmentation tasks
            acc = compute_accuracy(pred, labels)
            epoch_train_acc.append(acc)

            # IoU Calculation for Segmentation
            ious, mean_iou = calculate_iou(pred, labels, NUM_CLASSES)
            epoch_train_iou.append(mean_iou)
            for cls in range(NUM_CLASSES):
                epoch_train_iou_per_class[cls].append(ious[cls])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            print(
                f"Training - Epoch {epoch}, Batch {i}: Train Loss: {loss.item()}, Train Acc: {acc}, Train IoU {mean_iou}"
            )

        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_iou = []
        epoch_val_iou_per_class = {i: [] for i in range(NUM_CLASSES)}

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
                for cls in range(NUM_CLASSES):
                    epoch_val_iou_per_class[cls].append(ious[cls])

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
                os.path.join(
                    checkpoint_dir, f"segmentation_checkpoint_epoch_{epoch}.pth"
                ),
            )

        # Update logging lists for plotting later
        train_loss.append(np.mean(epoch_train_loss))
        test_loss.append(np.mean(epoch_val_loss))
        train_acc.append(np.mean(epoch_train_acc))
        test_acc.append(np.mean(epoch_val_acc))
        train_iou.append(np.mean(epoch_train_iou))
        test_iou.append(np.mean(epoch_val_iou))
        for cls in range(NUM_CLASSES):
            train_iou_per_class[cls].append(np.mean(epoch_train_iou_per_class[cls]))
            test_iou_per_class[cls].append(np.mean(epoch_val_iou_per_class[cls]))

        print(
            f"Epoch {epoch}: Train Loss: {train_loss[-1]}, "
            f"Val Loss: {test_loss[-1]}, "
            f"Train Acc: {train_acc[-1]}, "
            f"Val Acc: {test_acc[-1]}, "
            f"Train IoU: {train_iou[-1]}, "
            f"Val IoU: {test_iou[-1]}"
        )

    # Close the IoU log file
    iou_log_file.close()

    # Testing phase: Evaluate the model on the test dataset to measure its performance.
    model.eval()  # Set model to evaluation mode
    epoch_test_loss = []
    epoch_test_acc = []
    epoch_test_iou = []
    total_confusion_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES)
    )  # Initialize confusion matrix
    epoch_test_iou_per_class = {
        i: [] for i in range(NUM_CLASSES)
    }  # Initialize IoU per class tracking

    with torch.no_grad():
        for points, labels in test_dataloader:
            points, labels = points.to(device), labels.to(device)
            pred, _ = model(points)
            loss = criterion(pred, labels)
            epoch_test_loss.append(loss.item())

            # Compute accuracy
            acc = compute_accuracy(pred, labels)
            epoch_test_acc.append(acc)

            # Compute IoU
            ious, mean_iou = calculate_iou(pred, labels, NUM_CLASSES)
            epoch_test_iou.append(mean_iou)
            for cls in range(NUM_CLASSES):
                epoch_test_iou_per_class[cls].append(ious[cls])

            # Confusion matrix
            batch_confusion_matrix = compute_confusion_matrix(
                total_confusion_matrix, pred, labels, NUM_CLASSES
            )
            total_confusion_matrix += batch_confusion_matrix

    # After completing the loop over the test_dataloader
    average_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
    average_test_accuracy = sum(epoch_test_acc) / len(epoch_test_acc)
    average_test_iou = sum(epoch_test_iou) / len(epoch_test_iou)
    avg_test_iou_per_class = {i: [] for i in range(NUM_CLASSES)}
    for cls in range(NUM_CLASSES):
        avg_test_iou_per_class[cls].append(np.mean(epoch_test_iou_per_class[cls]))

    # Print the results
    print(
        f"Test Results - Loss: {average_test_loss:.4f}, Accuracy: {average_test_accuracy:.2f}, IoU: {average_test_iou:.2f}%"
    )

    # Plotting the results
    output_folder = FIGURES_DIRECTORY
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plot_losses(
        train_loss,
        test_loss,
        save_to_file=os.path.join(
            output_folder, "loss_plot" + str(NUM_POINTS) + ".png"
        ),
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

    class_names = [
        "Others",
        "Ground",
        "Vegetation",
        "Cars",
        "Trucks",
        "Powerlines",
        "Fences",
        "Poles",
        "Buildings",
    ]

    plot_iou_per_class(  # Training IoU plot
        train_iou_per_class,
        class_names,
        phase="Training",
        save_to_file=os.path.join(
            output_folder, "iou_train_per_class_plot" + str(NUM_POINTS) + ".png"
        ),
    )

    plot_iou_per_class(  # Testing IoU plot
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
        ),
    )

    # Save ious per class in train
    iou_class_file = open("train_iou_class.txt", "a")
    for cls, iou in train_iou_per_class.items():
        iou_class_file.write(f"{class_names[cls]}: iou = {iou[-1]:.4f}\n")
        iou_class_file.flush()
    iou_class_file.close()

    # Save ious per classin test
    iou_class_file = open("test_iou_class.txt", "a")
    for cls, iou in test_iou_per_class.items():
        iou_class_file.write(f"{class_names[cls]}: iou = {iou[-1]:.4f}\n")
        iou_class_file.flush()
    iou_class_file.close()


if __name__ == "__main__":
    main()
