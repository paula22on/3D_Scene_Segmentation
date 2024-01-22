import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyDataset  # TODO MODIFY IMPORT OF DATA
from model import MyModel  # TODO: MODIFY IMPORT OF MODEL
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

my_dataset = MyDataset()
dataloader = DataLoader()

my_model = MyModel().to(device)  # TODO: MODIFY MODEL NAME
criterion = ...  # TODO: ADD CRITERION
optimizer = ...  # TODO: ADD OPTIMIZER

loss_history = []

# TODO: MODIFY THIS LOOP TO FIT OUR DATA
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = my_model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

plt.plot(loss_history)
plt.title("Training loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()
