# PointNet 3D Scene Segmentation

## Overview

This project implements a deep neural network model using the PointNet architecture for classification and segmentation tasks on 3D point cloud data. The implementation supports both tasks with adjustable parameters for the number of points in the cloud, the number of classes, and the choice between using a weighted loss function or not.

### More about PointNet

PointNet is a pioneering deep learning architecture for processing unstructured 3D point cloud data. It's a powerful tool for various tasks in computer vision, robotics, and augmented reality, and this repository provides an implementation tailored for 3D scene segmentation.

## Features

- Classification and Segmentation: Support for both 3D point cloud classification and segmentation tasks.
- Weighted Loss Option: Option to use weighted loss during training to handle class imbalance.
- Custom Dataset Handling: Includes a custom dataset loader to handle 3D point cloud data efficiently.
- Visualization Tools: Functions to visualize training progress, including losses, accuracies, IoU scores, and confusion matrices.
- GPU Support: Utilizes GPU acceleration if available to speed up training and inference processes.

## Requirements

- Python 3.8 or later
- PyTorch 1.7.0 or later
- Matplotlib
- Numpy
- Pandas

## Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.8 or later installed.
3. Install the required Python packages:

```
pip install torch torchvision numpy pandas matplotlib tqdm
```

4. Navigate to the cloned repository's directory.

## Usage

To use this project for training or evaluating a PointNet model on your dataset, follow these steps:

Prepare your dataset in the required format and place it in the data directory.

Run the main script to start training or evaluation:

```
python main.py
```

### Customization options

You can customize the training process by modifying the following flags in main.py:

- SEGMENTATION: Set to True for segmentation tasks, or False for classification.
- WEIGHTED_LOSS: Enable or disable weighted loss calculation.
- NUM_POINTS: The number of points in each point cloud.
- NUM_CLASSES: The number of classes for classification/segmentation.

### Visualization

After training, use the visualization functions to analyze the performance:

```
plot_losses(train_loss, test_loss)
```

## Dataset

The dataset used in the experiments includes a large-scale aerial LiDAR dataset with over a 500M hand-labeled points spanning 10 km2 of area and eight object categories, known as DALES.

### Data pre-processing

To effectively manage a large dataset, downsampling is necessary. This project segments each sample into 100 sub-divisions using a dedicated script. Follow these steps for data pre-processing:

1. Download the original annotated LAS files from [this link](https://drive.google.com/file/d/1VKm05i-4fIi7xtws668LSmECbZTbvbEm/view) (~4G):

2. Place the downloaded dataset into a folder named dales_las. Inside the `dales_las` folder, ensure there are two sub-folders for train and test data:

```
dales_las/
  train/
  test/
```

3. Use the `data_preprocessing.py` script for data pre-processing. Run the script by specifying the path to the dataset and the divider values.

Example command:

```
python3 data_preprocessing.py /home/dales_las 10
```

In this example, `/home/dales_las` is the path to the dataset, and 10 is the divider value.

4. The script divides each sample into 100 cells (10 for the x-axis and 10 for the y-axis). Subsamples will be generated in .csv format and stored in a new `data/` folder, within the `train/` and `test/` sub-folders respectively.

```
data/
  train/
    <subsample>_0.csv
    <subsample>_1.csv
    <subsample>_2.csv
  test/
    <subsample>_0.csv
    <subsample>_1.csv
    <subsample>_2.csv
```

These steps will preprocess the dataset, making it suitable for further analysis and modeling.

To visualize the subsamples, just run the `data_visualization.py` script by specifying the path to the data subsample.

Example command:

```
python3 data_visualization.py data/train/10_divisions_0.csv
```

## Training

Explain the training process, including hyperparameters, data augmentation techniques, and any other important details. Provide examples of how to train the model with different settings.

## Evaluation

Describe how to evaluate the model's performance on a validation or test dataset. Include metrics used for evaluation and how to interpret the results.

## Results

Share the results of your experiments, including any visualizations or plots. Compare the performance of your model with state-of-the-art methods if applicable.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Authors

- [laurahomet](https://github.com/laurahomet) - Laura Homet
- [albertpedra45](https://github.com/albertpedra45) - Albert Pedraza
- [paula22on](https://github.com/paula22on) - Paula Osés
- [dom27d](https://github.com/dom27d) - Daniel Ochavo
