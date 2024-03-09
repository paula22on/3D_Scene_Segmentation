# PointNet 3D Scene Segmentation

Final project for the 2023-2024 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by Paula Osés, Albert Pedraza, Laura Homet, Daniel Ochavo.

Advised by Mariona Carós.


## Table of contents

- [Overview](#overview)
  - [Project features](#project-features)
  - [More about PointNet](#more-about-pointnet)
- [DALES dataset](#dales-dataset)
  - [Downsampling](#downsampling)
    - [Random sampling](#random-sampling)
  - [Data balancing](#data-balancing)
  - [Data rotation](#data-rotation)
- [Architecture (PointNet)](#architecture-pointnet)
  - [First test with ShapeNet dataset](#first-test-with-shapenet-dataset)
  - [Layer upsize](#layer-upsize)
- [Results](#results)
  - [First approach (case study 1)](#first-approach-case-study-1)
  - [Model improvements](#model-improvements)
    - [Weighted loss](#weighted-loss)
    - [Hyperparameters](#hyperparameters)
    - [Sample rotation in batch](#sample-rotation-in-batch)
  - [Second approach (case study 2)](#second-approach-case-study-2)
  - [Third approach (case study 3)](#third-approach-case-study-3)
  - [Final results (best case)](#final-results-best-case)
- [How to](#how-to)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [How to prepare the dataset from scratch](#how-to-prepare-the-dataset-from-scratch)
    - [Download](#download)
    - [Data-preprocessing](#data-preprocessing)
  - [How to train the model](#how-to-train-the-model)
    - [Setting the environment in Google Cloud?](#setting-the-environment-in-google-cloud)
    - [Running training scripts](#running-training-scripts)
      - [Customization options](#customization-options)
      - [Visualization](#visualization)
  - [How to evaluate the model](#how-to-evaluate-the-model)
    - [Running the evaluation scripts](#running-the-evaluation-scripts)
    - [Interpreting the results](#interpreting-the-results)
- [Conclusions](#conclusions)
- [Future work](#future-work)


## Overview

This project implements a deep neural network model using the PointNet architecture for classification and segmentation tasks on 3D point cloud data. The implementation supports both tasks with adjustable parameters for the number of points in the cloud, the number of classes, and the choice between using a weighted loss function or not.

- [ ] **TODO:** Ampliar això com a introducció/motivació del projecte

### Project features

- Classification and Segmentation: Support for both 3D point cloud classification and segmentation tasks.
- Weighted Loss Option: Option to use weighted loss during training to handle class imbalance.
- Custom Dataset Handling: Includes a custom dataset loader to handle 3D point cloud data efficiently.
- Visualization Tools: Functions to visualize training progress, including losses, accuracies, IoU scores, and confusion matrices.
- GPU Support: Utilizes GPU acceleration if available to speed up training and inference processes.

### More about PointNet

PointNet is a pioneering deep learning architecture for processing unstructured 3D point cloud data. It's a powerful tool for various tasks in computer vision, robotics, and augmented reality, and this repository provides an implementation tailored for 3D scene segmentation.


## DALES dataset

- [ ] **TODO:** Explicar DALES i tot el preprocessing i posar visualitzacions
- [ ] **TODO:** Add whole DALES dataset image

### Downsampling
- [ ] **TODO:** Add image of divided tiles

#### Random sampling
- [ ] **TODO:** Add image of original subsample VS random sample

### Data balancing
- [ ] **TODO:** Add image of original subsample VS random sample VS balanced sample

### Data rotation
- [ ] **TODO:** Add image of balanced sample VS rotated 45 degrees VS rotated 90 degrees


## Architecture (PointNet)
- [ ] **TODO:** Explicar architecture
- [ ] **TODO:** Add image of PointNet architecture

### First test with ShapeNet dataset
- [ ] **TODO:** Grafics de la shapenet, resultats en plots, confusion matrix, T-net
- [ ] **TODO:** Add image of ShapeNet dataset

### Layer upsize
- [ ] **TODO:** Explain potential size changes to originial PointNet

## Results

- [ ] **TODO:** Complete this section. Explain the training process, including hyperparameters, data augmentation techniques, and any other important details. Provide examples of how to train the model with different settings.

- [ ] **TODO:** Share the results of your experiments, including any visualizations or plots. Compare the performance of your model with state-of-the-art methods if applicable.

- [ ] **TODO:** Describe how to evaluate the model's performance on a validation or test dataset. Include metrics used for evaluation and how to interpret the results.

### First approach (case study 1)
- [ ] **TODO:** Add plots and images of first results (the one that only predicted majority classes)

### Model improvements
#### Weighted loss
- [ ] **TODO:** Add formula

#### Hyperparameters
#### Sample rotation in batch

### Second approach (case study 2)
- [ ] **TODO:** Add plots and images of the results

### Third approach (case study 3)
- [ ] **TODO:** Add plots and images of the results

### Final results (best case)
- [ ] **TODO:** Add plots and images of the results


## How to

### Requirements
- Python 3.8 or later
- PyTorch 1.7.0 or later
- Matplotlib
- Numpy
- Pandas

### Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.8 or later installed.
3. Install the required Python packages:

```
pip install torch torchvision numpy pandas matplotlib tqdm
```

4. Navigate to the cloned repository's directory.

### How to prepare the dataset from scratch

The dataset used in the experiments includes a large-scale aerial LiDAR dataset with over a 500M hand-labeled points spanning 10 km2 of area and eight object categories, known as DALES.

#### Download

1. Download the original annotated LAS files from [this link](https://drive.google.com/file/d/1VKm05i-4fIi7xtws668LSmECbZTbvbEm/view) (~4G):

2. Place the downloaded dataset into a folder named dales_las. Inside the `dales_las` folder, ensure there are two sub-folders for train and test data:

```
dales_las/
  train/
  test/
```

#### Data-preprocessing

To effectively manage a large dataset, downsampling is necessary. This project segments each sample into 100 sub-divisions using a dedicated script.

Use the `data_preprocessing.py` script for data pre-processing. Run the script by specifying the path to the dataset and the divider values.

Example command:

```
python3 data_preprocessing.py /home/dales_las 10
```

In this example, `/home/dales_las` is the path to the dataset, and 10 is the divider value.

The script divides each sample into 100 cells (10 for the x-axis and 10 for the y-axis). Subsamples will be generated in .csv format and stored in a new `data/` folder, within the `train/` and `test/` sub-folders respectively.

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

- [ ] **TODO:** Make sure this is up to date. Data visualization is not a PY script anymore, but we can add it somewhere.


### How to train the model

#### Setting the environment in Google Cloud?

- [ ] **TODO:** Is this needed?


#### Running training scripts

To use this project for training or evaluating a PointNet model on your dataset, follow these steps:

Prepare your dataset in the required format and place it in the data directory.

Run the main script to start training or evaluation:

```
python main.py
```

#### Customization options

You can customize the training process by modifying the following flags in main.py:

- SEGMENTATION: Set to True for segmentation tasks, or False for classification.
- WEIGHTED_LOSS: Enable or disable weighted loss calculation.
- NUM_POINTS: The number of points in each point cloud.
- NUM_CLASSES: The number of classes for classification/segmentation.

#### Visualization

After training, use the visualization functions to analyze the performance:

```
plot_losses(train_loss, test_loss)
```

- [ ] **TODO:** Add a script for visualization?


### How to evaluate the model
#### Running the evaluation scripts
#### Interpreting the results

## Conclusions

- [ ] **TODO:** Complete this section

## Future work

- [ ] **TODO:** Complete this section



## Authors

- [laurahomet](https://github.com/laurahomet) - Laura Homet
- [albertpedra45](https://github.com/albertpedra45) - Albert Pedraza
- [paula22on](https://github.com/paula22on) - Paula Osés
- [dom27d](https://github.com/dom27d) - Daniel Ochavo
