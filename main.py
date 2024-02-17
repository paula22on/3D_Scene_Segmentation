import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import ClassificationPointNet, SegmentationPointNet
from utils import compute_accuracy

SEGMENTATION = True
NUM_POINTS = 2048

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def import_dataset(split = "train"):
    dataset = []
    path = "data/" + split + '/'
    tiles = os.listdir(path)
    for tile in tiles:
        samples = os.listdir(path + tile + '/')
        print("Loading currently tile: " + tile)
        for x in samples:
            item = pd.read_csv(path + tile + '/' + x, dtype=int)
            dataset.append(item)
    return dataset

#Loading time...2m
train_dataset = import_dataset()
test_dataset = import_dataset(split = "test")
#---- All above code works! Currently testing...
train_dataset = MyDataset(train_dataset, NUM_POINTS, "train")
test_dataset = MyDataset(test_dataset, NUM_POINTS, "test")
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [2300, 600])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


if SEGMENTATION:
    model = SegmentationPointNet(num_classes=6, point_dimension=3)
else:
    model = ClassificationPointNet(num_classes=16, point_dimension=3, segmentation = False)


criterion = torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint_dir = "checkpoints-segmentation"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Training and Evaluation Loop
epochs = 80
train_loss = []
val_loss = []
test_loss = []
train_acc = []
val_acc = []
test_acc = []
best_loss = np.inf

for epoch in tqdm(range(epochs)):
    model.train()
    epoch_train_loss = []
    epoch_train_acc = []

    # Training Loop
    for i, data in enumerate(train_dataloader):
        points, labels = data
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred, _ = model(points)

        labels = labels - 1
        loss = criterion(pred.view(-1, 6), labels.view(-1))
        epoch_train_loss.append(loss.item())

        # Accuracy Calculation for Segmentation
        acc = compute_accuracy(pred, labels)
        epoch_train_acc.append(acc)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(
            f"Training - Epoch {epoch}, Batch {i}: Train Loss: {loss.item()}, Train Acc: {acc}"
        )

    epoch_val_loss = []
    epoch_val_acc = []

    # Validation Loop
    with torch.no_grad():
        for data in val_dataloader:
            points, labels = data
            points, labels = points.to(device), labels.to(device)
            pred, _ = model(
                points, segmentation=SEGMENTATION
            )  #! TEST IF THIS WAY OF CALLING THE FORWARD WORKSS
            labels = labels - 1
            loss = criterion(pred.view(-1, 6), labels.view(-1))
            epoch_val_loss.append(loss.item())

            acc = compute_accuracy(pred, labels)
            epoch_val_acc.append(acc)

        print(
            "Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f"
            % (
                epoch,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_val_loss), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_val_acc), 4),
            )
        )

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
    val_loss.append(np.mean(epoch_val_loss))
    train_acc.append(np.mean(epoch_train_acc))
    val_acc.append(np.mean(epoch_val_acc))

    print(
        f"Epoch {epoch}: Train Loss: {train_loss[-1]}, "
        f"Val Loss: {val_loss[-1]}, "
        f"Train Acc: {train_acc[-1]}, "
        f"Val Acc: {val_acc[-1]}"
    )

# Testing loop
model.eval()
epoch_test_loss = []
epoch_test_acc = []

with torch.no_grad():
    for points, labels in test_dataloader:
        points, labels = points.to(device), labels.to(device)
        pred, _ = model(points)
        labels = labels - 1
        loss = criterion(pred.view(-1, 6), labels.view(-1))
        epoch_test_loss.append(loss.item())

        acc = compute_accuracy(pred, labels)
        epoch_test_acc.append(acc)

# After completing the loop over the test_dataloader
average_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
average_test_accuracy = sum(epoch_test_acc) / len(epoch_test_acc)

# Print the results
print(
    f"Test Results - Loss: {average_test_loss:.4f}, Accuracy: {average_test_accuracy:.2f}%"
)

# Logging for testing
test_loss.append(np.mean(epoch_test_loss))
test_acc.append(np.mean(epoch_val_acc))

# Plotting the results

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
