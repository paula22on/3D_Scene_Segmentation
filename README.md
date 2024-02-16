# PointNet 3D Scene Segmentation

## Overview

This repository contains the implementation of PointNet for 3D scene segmentation. PointNet is a deep learning model designed for processing and segmenting 3D point cloud data, making it useful for tasks like object recognition and scene segmentation in 3D environments.

## Table of Contents

Introduction
Installation
Usage
Dataset
Training
Evaluation
Results
Contributing
License

## Introduction

PointNet is a pioneering deep learning architecture for processing unstructured 3D point cloud data. It's a powerful tool for various tasks in computer vision, robotics, and augmented reality, and this repository provides an implementation tailored for 3D scene segmentation.

## Installation

To get started, follow these steps to set up the project:

xxx

## Usage

Describe how to use your code. Provide clear instructions and examples on how to run the training, testing, and inference processes.

xxx

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
