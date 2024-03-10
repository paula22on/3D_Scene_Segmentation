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
    - [Running training scripts](#running-training-scripts)
      - [Customization options](#customization-options)
      - [Visualization](#visualization)
  - [How to evaluate the model](#how-to-evaluate-the-model)
    - [Running the evaluation scripts](#running-the-evaluation-scripts)
    - [Interpreting the results](#interpreting-the-results)
- [Conclusions](#conclusions)
- [Future work](#future-work)


## Overview

Within the field of computer vision, the ability to accurately analyze 3 dimensional data stands as a cornerstone for nummerous applications, such as autonomous driving, VR or urban planning. In order to satisfy these, appears the need of efficient methods to classify and segment this data, enabling machines to understand and interact with complex 3D environments in a meaningful way.

This project makes use of the PointNet architecture, a deep neural network designed to tackle these challenges providing a direct analysis of 3D point clouds. Our implementation leverages the segmentation task of this framework and enriches it by enabling customizable parameters that will adjust the architecture to the demands of the dataset.

### Project features

- Classification and Segmentation: Support for both 3D point cloud classification and segmentation tasks.
- Weighted Loss Option: Option to use weighted loss during training to handle class imbalance.
- Custom Dataset Handling: Includes a custom dataset loader to handle 3D point cloud data efficiently.
- Visualization Tools: Functions to visualize training progress, including losses, accuracies, IoU scores, and confusion matrices.
- GPU Support: Utilizes GPU acceleration if available to speed up training and inference processes.

### More about PointNet

PointNet is a pioneering deep learning architecture for processing unstructured 3D point cloud data. It's a powerful tool for various tasks in computer vision, robotics, and augmented reality, and this repository provides an implementation tailored for 3D scene segmentation.


## DALES dataset

The dataset used for this project is DALES, or Dayton Annotated Laser Earth Scan dataset. It is a new large-scale aerial LiDAR data set with nearly a half-billion points spanning 10 square kilometers of area. Unlike other datasets, DALES focuses on aerially collected data, introducing unique challenges and opportunities for 3D urban modeling and large-scale surveillance applications. With its vast scale and high resolution, DALES provides an extensive base for evaluating and developing 3D deep learning algorithms.


DALES contains forty scenes of dense, labeled aerial data spanning multiple scene types, including urban, suburban, rural, and commercial. The data was hand-labeled by a team of expert LiDAR technicians into eight categories: ground, vegetation, cars, trucks, poles, power lines, fences, and buildings. The entired dataset is split into testing and training, and provided in 3 different data formats. The data format used in this project is LAS (LiDAR Aerial Survey).

![image](https://github.com/paula22on/3D_Scene_Segmentation/assets/135391540/db7a0824-a3d4-4459-bc0f-6f4104e9605c)


Each point in this dataset has been hand-labeled under 9 different categories:

| Category | Color |
| -------- | ----- |
| Ground | 'blue' |
| Vegetation | 'green' |
| Power Lines | 'yellow' |
| Poles | 'pink' |
| Buildings | 'red' |
| Fences | 'gray' |
| Trucks | 'orange' |
| Cars | 'purple' |
| Unknown | 'black' |

In order to prepare the dataset for use with the PointNet architecture, we tailored a preprocessing pipeline for the DALES dataset, designed to enhance the quality and usability of the dataset, optimizing it for effective training and evaluation of our model. The key steps in this process are the following:

1. **Simplification.** We begin by simplifying the raw LAS (LiDAR Aerial Survey) files, extracting essential data such as X, Y, Z coordinates, and classification labels. This step distills the dataset down to its most important features for efficient processing.

2. **Normalization.** To ensure uniformity across the dataset, we normalize the X and Y coordinates. This involves adjusting these coordinates to a common scale by subtracting the minimum value found in the dataset, ensuring that our model trains on consistently scaled data.

3. **Subsampling.** Given the extensive size of the DALES dataset, direct processing is computationally challenging. To address this, we implement a subsampling strategy, dividing the point cloud into smaller, more manageable regions. This step allows us to maintain a high level of detail while reducing the computational load. The divider parameter controls the granularity of this subdivision, enabling flexibility in the balance between detail and performance.

4. **Balancing Classes.** Imbalances in the distribution of classes within the dataset can bias the model's training process. To mitigate this, we optionally apply a class balancing step to ensure that each class is equally represented, enhancing the model's ability to generalize across different categories.

5. **Data Augmentation - Rotation.** To improve the robustness of our model, we introduce variability into the training data through random rotations about the Z-axis. This form of data augmentation simulates a wider variety of scenarios, helping the model learn to recognize structures from different orientations.

6. **Exporting Processed Data.** After preprocessing, the data is exported into CSV files, organized by sample type (train or test) and further divided based on the subsampling strategy. This structured format makes it easy to manage and access the data during training and evaluation phases.

The image below depicts the top view of one of the forty scenes provided by DALES, covering an area of 250 square meters. Each point in the point cloud is colored according to its assigned color as described in the presented table.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/7f791a73-7785-4d94-8953-0f9f3b45c41f" width="800" alt="cm">

We computed the ditribution of points across the complete dataset, for each category, diven the train/tast partition that was present on the original data. 

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/981c1c15-2723-44de-8cbd-69052bf0a29f" width="800" alt="cm">

All the following methods and results, have the awareness that there are 3 majority classes (ground, vegetation, buidlings) and other minority classes that may be more difficult to learn and predict correctly.

### Subsampling

Given the computational challenges inherent in processing an entire scene of the dataset, and considering that the model operates on a limited number of points, we cannot effectively capture all the relationships and dependencies among every point in the point cloud. Thus, there is a necessity to partition the scenes into subsamples. Through experimentation, we have determined that dividing each scene into 100 pieces strikes a balance between reducing the number of points while preserving the essential relationships among them.

This division process considers only the X and Y dimensions, leaving the Z dimension untouched. The Z dimension represents the vertical axis of a scene, and dividing it would result in the loss of semantic information. The X and Y axes are each divided by a factor of 10. Consequently, an entire tile of the DALES dataset is partitioned into 100 pieces.

The image below illustrates the top left corner of the first scene in DALES.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/36137633/bff89082-2da4-475e-8571-d60c9c2524d7" width="800" alt="original_tile_top_left_corner">

This area is processed and subdivided into 5 distinct samples, as depicted below.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/36137633/d42c17bb-956e-4267-b3fe-18aa01e3c28a" width="800" alt="original_samples_0_to_4">

#### Random sampling

As part of the subsampling process, each sample is further reduced through random subsampling of points. This random sampling technique is crucial for reducing the computational burden on the model, as it processes only a subset of points from the point cloud. By selecting randomly N amount of points for each sample, where N is typically configured to 2048 points per sample in this project, we ensure efficient resource utilization while retaining crucial spatial information vital for accurate segmentation.

Below, we present a comparison between an original subsample (a single division of the original tile) containing a large number of points and a subsample where 2048 points have been randomly selected.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/36137633/3bd77f76-2fd3-4601-be91-b2b91a8d1831" width="800" alt="original_vs_random_subsample_0">

### Data balancing

Class imbalance in point clouds poses a challenge as it can lead to biased model predictions and reduced performance on minority classes. In our dataset samples, we observe significant class imbalance, with an abundance of points for majority classes like ground, vegetation, and buildings, compared to minority classes such as poles, power lines, and fences. 

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/36137633/5fe62d8c-bc31-41ef-a8b3-f09432dcbf43" width="800" alt="class_imbalance_subsample_0">

To mitigate this issue, we address class imbalance by modifying training samples. For each class, we compute the average number of points. Subsequently, if the number of points for a class is below the average, we replicate a random point of that class to augment the sample. Otherwise, if the number of points exceeds the average, we randomly select N points, where N is the average.

The combination of the random sampling method described earlier with this balance, results in the representation we showcase below. The figure illustrates the original subsample on the left and the balanced subsample after the selection of 2048 points on the right.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/36137633/c632e38c-3d21-446e-8769-a972288c90a6" width="800" alt="original_vs_balanced_subsample_0">

### Data rotation

In a model that deals with point clouds, data augmentation is crucial for several reasons. Firstly, it helps in enhancing the model's generalization ability by exposing it to a wider variety of input variations. Secondly, it aids in addressing the problem of limited data, which is often encountered in real-world scenarios. Additionally, data augmentation assists in mitigating overfitting by introducing variability into the training data.

PointNet is a model invariant to permutations of 3D point cloud figures, which is why data augmentation becomes especially pertinent. Since the model remains unaffected by the order of points, we opt for augmentation techniques like rotation to enrich the dataset. Rotation allows the model to learn from various viewpoints of the same object, thereby improving its robustness and accuracy.

The rotation of points occurs around the Z-axis by either a random angle or a specified angle if provided. In this context, the rotation operation affects only the X and Y coordinates of the points while keeping the Z coordinate unchanged. This rotation is implemented by applying a 2D rotation matrix to each point, as follows:

$$
\begin{align*}
X' & = X \cos(\theta) - Y \sin(\theta) \\
Y' & = X \sin(\theta) + Y \cos(\theta) \\
Z' & = Z
\end{align*}
$$

Here, theta represents the rotation angle in degrees. The rotated points P' are obtained by multiplying the original points P by the rotation matrix. This transformation effectively rotates the points around the origin (0,0) in the XY plane by the specified angle theta, allowing for various orientations of the point cloud data.

The figure below illustrates a comparison between the original sample and the rotated sample rotated by 45 degrees.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/36137633/5a8d6be6-8e50-463c-910d-64bed0eb5cc9" width="800" alt="original_vs_rotated_subsample_0">

## Architecture (PointNet)
PointNet is a pioneering deep neural network designed specifically for processing 3D point clouds, which are collections of data points defined in a three-dimensional coordinate system. Developed by researchers at Stanford University in 2016, PointNet stands out as the first neural network architecture to directly work with 3D point clouds, eliminating the need for pre-processing steps such as voxelization or rendering. This innovative approach enables PointNet to efficiently learn both the global and local features of point clouds, making it highly effective for a wide range of applications, including object classification, part segmentation, and scene semantic parsing.

The architecture of PointNet is ingeniously structured to cater to the unique challenges posed by point cloud data. It comprises two main components: a classification network and a segmentation network. The classification network is tasked with assigning a classification score for each of the predefined classes, enabling it to identify the type of object represented by the point cloud. On the other hand, the segmentation network combines global and local features to output per-point scores, which are essential for understanding the detailed structure of objects and their constituent parts.

Several key innovations make PointNet exceptionally effective in handling point clouds:

- **Spatial Transformation Network (T-Net)**: PointNet introduces a T-Net component that ensures invariance to geometric transformations. This means that the network can recognize objects regardless of their orientation, scale, or position in space, addressing a common challenge in 3D object recognition.

- **Permutation Invariance**: Given that point clouds are inherently unordered, PointNet leverages a symmetric function, specifically max pooling, to ensure that its output is invariant to the order of the input points. This is crucial for processing point clouds directly without needing to impose an artificial order on the data.

- **Local and Global Feature Aggregation**: PointNet captures the intricate details of point clouds by effectively merging local and global features. This allows the network to recognize fine-grained patterns and structures within the data, facilitating accurate segmentation and classification.

- **Efficiency and Scalability**: By directly processing point clouds without the need for complex pre-processing, PointNet achieves high efficiency and scalability. This makes it suitable for handling large-scale point cloud data, which is common in applications such as autonomous driving, robotics, and 3D modeling.

The complete architecture can be seen in the following image,
![image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*lFnrIqmV_47iRvQcY3dB7Q.png)

### First test with ShapeNet dataset
We initiated our exploration of the PointNet architecture with a straightforward classification task, utilizing the ShapeNetCore dataset from the larger ShapeNet collection. This subset offers a well-curated selection of single, clean 3D models, each manually verified for category accuracy and alignment, spanning 55 common object categories with around 51,300 unique models. After conducting training over 80 epochs, we achieved an accuracy of 92.2% and a loss of 0.06. Below, we present the confusion matrix for further insight into the model's performance:

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/117265561/4a39e610-2d8a-4ad1-a80f-7def538fa4c6" width="500" alt="cm">

In addition to evaluating the PointNet architecture's classification capabilities using the ShapeNetCore dataset, this testing phase also served to deepen our understanding of the T-Net component within PointNet. The T-Net, crucial for ensuring the model's invariance to geometric transformations of the input point clouds, was closely examined. To illustrate the transformations learned and applied by the T-Net, we've included a series of images showcasing these adjustments. These visual representations not only highlight the T-Net's functionality but also demonstrate its impact on the model's ability to accurately interpret and classify 3D shapes.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/117265561/7dd84a8f-4b6f-4878-a530-518a4064561d" width="300" alt="cm">  <img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/117265561/bf2259a1-1f73-40e8-901e-75380f4fe4a1" width="300" alt="cm"> 
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/117265561/fdc6f2e8-bedc-4726-90b6-9cde9b4cfd54" width="300" alt="cm"> <img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/117265561/d906c4e0-584b-49d2-9a9d-a533da07ee8b" width="300" alt="cm"> 

### Layer upsize

For simpler tasks, such as the classification challenge we undertook with the ShapeNetCore dataset, the PointNet architecture can be effectively streamlined by reducing the number of layers. This modification maintains high accuracy while optimizing computational efficiency. Conversely, for more complex applications, such as segmentation tasks that demand a finer understanding of point cloud data, the architecture can be scaled up by increasing the number of layers. This flexibility allows PointNet to be tailored to a wide range of demands, balancing performance and computational resources according to the specific requirements of the task at hand.

## Results

We are training the dataset with a PointNet model using an NLLLoss or the Inverse Weighted Cross Entropy Loss. For the batch size, using 32 as batch size is expected for this kind of problem where literature expects not a great number of batch sizes for preventing overfitting. 

We also set an Adam optimizer with a learning rate of 0.001. A number of 80 epochs is a suitable number for this kind of project, as an optimal loss is not expected, only a good generalization of the task.

We are evaluating our model with the accuracy metric, iou metric per class, mean iou, and the confusion matrix. 

Our validation is the same as evaluation but during the training loop, only for validation data. A remark must be done, the augmentation process of rotating data for every epoch, is done on training and validation data. 

### Naive approach 
Training on the original dataset without any changes. The model only predicts 3 classes out of 9. No augmentation or loss method is performed so the majority classes in terms of distribution have all the impact for the model. 
We can start to point out that the model keeps the structure of the city merely intact, but the majority classes are the only ones predicted for the moment.

Our test results are the following:

Test Results - Loss: 0.6857, Accuracy: 0.75, IoU: 0.19%. We may consider it to fail in terms of the global semantic segmentation task.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/0edb5f8e-4679-4ba4-88f1-56e90322260d" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/207c6b11-f766-43fc-836e-7399f3db2718" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/40960eef-aabb-4383-809a-132321368201" width="500" alt="cm">

### Data balancing + Augmentation 
For the next phase, we are using balancing on the original dataset during preprocessing on training data. Also we are augmenting our training data by performing random rotation on the pre-processing module, as it would give us an advantage over more training data.
We also made re-ajustment on the model architecture, in terms of upsizing the layers of the architecture, in order to be more suitable for our task, as exlained above. Upsizing occurs mostly inside the T-Net layers.

Our test results are the following:

Test Results - Loss: 1.2387, Accuracy: 0.63, IoU: 0.17%. In this case we are indeed predicting nearly all classes for our task, which means that some following tunning can be performed, but the results are presented in a positive manner.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/d6b84bc9-b06b-47ea-b4ff-8532e106a41f" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/f77abd50-ab05-4502-879f-8513ec5c29b0" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/08f0d646-efe0-45dd-8cc3-ae43b78a5a21" width="500" alt="cm">

### Inverse Weighted Loss
For the sake of investigation, we also tried to train our model using the inverse weighted loss on the original data, (without any balancing nor augmentation). It performs positively for non majority distributed classes, but it lacks coherence in terms of prediction. At this point our model doesn't predict neither vegetation or ground correctly, which obviously discards the model. 

We also performed random rotation for every epoch, as means of generalizing best our data. It did improve our training process, but it wasn't enough for our model to get good results for both minority and majority classes. 

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/a7c473a4-565d-428d-9b31-f3dce499ab79" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/e6f21523-9fe1-4608-aded-9326059be1fb" width="500" alt="cm">

Our test results are the following: 

Test Results - Loss: 1.2973, Accuracy: 0.56, IoU: 0.15%.

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/f18e2fbc-fadf-45e9-ae87-64d74926918d" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/15eba654-2576-4e24-95e5-4f1f83a933f6" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/12626df7-d850-4e3b-b286-92757cb2ccc2" width="500" alt="cm">



### Final results (best case)

For the best model, we gathered the best performing method, which is balancing + aumentation on the preprocessing module, in order to amplify our train dataset. Then we also used the random rotation on every epoch, in terms of trying to generalize our learning better.

We are selecting this model as our best performance model, as it incroporates the best methods/processes that we tried in order for the architecure to learn semantic segmenation. We can also select it by its performance on the mean IoU across majority and minority labels, as we rate performance higher on minority classes as long as majority classes don't lose coherence. 

<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/da8288b7-65ef-4044-92d3-207e6e24e096" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/c28f799d-033c-45fd-82bb-f5104f98b4d1" width="500" alt="cm">


Our test results are the following:

Test Results - Loss: 1.2624, Accuracy: 0.60, IoU: 0.18%


<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/a061828d-6cef-4477-885c-83debbc4586a" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/e3b88b5b-7275-44f5-88ce-71817c76867a" width="500" alt="cm">
<img src="https://github.com/paula22on/3D_Scene_Segmentation/assets/55758205/1b3f6a93-b793-4c4b-8d3a-566a3c25a267" width="500" alt="cm">



## How to

### Requirements
Before starting, ensure your system meets the following requirements:
- [ ] **TODO:** MISSING UPDATED REQUIREMENTS.TXT!!
- Python 3.8 or later
- PyTorch 1.7.0 or later
- Matplotlib
- Numpy
- Pandas

### Installation

**Step 1: Clone the Repository**

To get started, clone the repository to your local machine using the command: 

```
git clone https://github.com/paula22on/3D_Scene_Segmentation.git
```

**Step 2: Set Up Python Environment**

Ensure you have Python 3.8 or later installed on your system. You can verify your Python version by running:

```
python3 --version
```

If you do not have Python installed, download it from [python.org](https://www.python.org/) and follow the installation instructions for your operating system.

**Step 3: Create a Virtual Environment**

Navigate to the cloned repository directory:

```
cd <repository>
```

Create a virtual environment named venv (or any name you prefer) by running:

```
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  
```
venv\Scripts\activate
```

- On macOS/Linux:

```
source venv/bin/activate
```

**Step 4: Install Required Packages**

Install all the required packages listed in the requirements.txt file:

```
pip install -r requirements.txt
```

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

To effectively manage a large dataset, downsampling is necessary. This project segments each sample into 100 sub-divisions using a dedicated script (data_preprocessing.py). This scripts also incorporates options for class balancing and applying randome rotation for data augmentation.

Use the script with the following arguments:
- `dir_path`: Path to the directory containing the DALES dataset in .las format.
- `divider`: Number used for subsampling the dataset.
- `--balance` (optional): Flag to balance the classes in the dataset.
- `--rotate` (optional): Flag to apply random rotation around the Z-axis for the training data.

Example command:

```
python3 data_preprocessing.py /home/dales_las 10 --balance --rotate

```
Here, /home/dales_las is the dataset directory path, 10 is the divider value for subsampling, --balance enables class balancing, and --rotate applies random rotations to the training data.

The script processes each LAS file in the train and test directories. Subsamples are generated in .csv format and stored in a new data/ folder, within train/ and test/ subfolders:

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

### How to train the model

To use this project for training the PointNet model on your dataset, prepare your dataset in the required format and place it in the data directory. Then run the main script to start training:

```
python main.py
```

#### Customization options

You can customize the training process by modifying the following flags in main.py:

- SEGMENTATION: Set to True for segmentation tasks, or False for classification.
- WEIGHTED_LOSS: Enable or disable weighted loss calculation.
- NUM_POINTS: The number of points in each point cloud.
- NUM_CLASSES_SEGMENTATION: The number of classes for segmentation.
- NUM_CLASSES_CLASSIFICATION: The number of classes for classification.
- RANDOM_ROTATION_IN_BATCH: Set to True if you want to doa random rotation in each batch, otherwise, False.
- CHECKPOINT_DIRECTORY: The directory to save the checkpoint.
- FIGURES_DIRECTORY: The directory to save the resulting figures.

You can modify the different parameters in the config.py file:

```
SEGMENTATION = True
WEIGHTED_LOSS = False
NUM_POINTS = 2048
NUM_CLASSES_SEGMENTATION = 9
NUM_CLASSES_CLASSIFICATION = 16
RANDOM_ROTATION_IN_BATCH = True
CHECKPOINT_DIRECTORY = "evaluation/checkpoints-segmentation"
FIGURES_DIRECTORY = "evaluation/figures"
```
Once the file is saved with you desired parameters you can run the main script using the command above.

#### Visualization

After training, use the visualization functions to analyze the performance, the figures will be saved in the directory specified in the config.py file.

List of visualizations:
- Loss graph
- Accuracy graph
- IoU
- IoU per class

In this segment of the code you can see the name and location for each plot:

```
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

    plot_iou_per_class(  # Training IoU plot
        train_iou_per_class,
        class_names,
        phase="Training",
        save_to_file=os.path.join(
            output_folder, "iou_train_per_class_plot" + str(NUM_POINTS) + ".png"
        ),
    )

```

### How to evaluate the model

#### Running the evaluation scripts

Once the model was trained, we stored the model checkpoints for future evaluation and uploaded it into a downloadable site. 

The evaluation scripts are placed within the `evaluation/` directory. To evaluate the model and the different approaches we took for training, follow these steps:

1. Download the checkpoints from [this link](https://drive.google.com/file/d/1fsB9gWDaHmKFjCH1FpFEHbD1RPLz-kSr/view?usp=drive_link)
2. Unzip the `segmentation-checkpoints.zip` file
3. Place the checkpoints inside the `evaluation/` directory. It is impotant to keep the folders and naming consistent with this directory structure:

  ```
  evaluation/
    checkpoints-segmentation/
      segmentation_checkpoint_naive.pth
      segmentation_checkpoint_augmentation.pth
      segmentation_checkpoint_rotated.pth
      segmentation_checkpoint_weighted.pth
  ```

4. Each of the four approaches presented corresponds to a checkpoint. We use arguments to evaluate each of the checkpoints, as follows:

  - `--naive`: Evaluation of the naive aproach.
  - `--augmentation`: Evaluation of a model trained with data balancing and rotation.
  - `--rotated`: Evaluation trained with a model trained with rotated data.
  - `--weighted`: Evaluation of a model trained with weighted inverse loss.

5. Navigate to the `evaluation` directory and run the evaluation script with one of these arguments to evaluate the corresponding training scenario.

Example command:

```
cd evaluation/
python3 evaluation.py --rotated

```

#### Interpreting the results

After execution, various plots and 3D point cloud figures get stored within the `figures` directory.

- `confmatrix_plot2048_rotated.png`: Plot of the confusion matrix resulting from trianing with the specified configuration.

- `iou_per_test_class_plot2048_rotated.png`: Plot of the confusion matrix resulting from trianing with the specified configuration.

- `sample_data/`: This directory contains a list of original and predicted samples in CSV format. Both the original and predicted samples are presented in two forms:
  - `original_sample_2048points_0.csv`: This is the normal CSV format of a sample, where each point of the sample is lsited with x, y, z, and label values.
  - `original_sample_2048points_0_to_99.csv`: This is a CSV containing the point values for 100 concurrent samples. All points are also listed as x, y, z, and label values.

- `sample_images/`: This directory contains images of original and predicted samples in PNG format. Both the original and predicted samples are presented in two forms:
  - `original_sample_2048points_0.png`: This file contains a normal 3D plot of a sample point cloud.
  - `original_sample_2048points_0_to_99.png`: This file displays a top-view plot, excluding the Z-axis, depicting 100 consecutive samples concatenated across the X-Y area.


## Conclusions

During this project, we explored the PointNet architecture with the classificaction and the semantic segmentation task. A big challenge was accomplished, that is training a pointcloud model using a large dataset model, specially on the semantic segmentation task. Other challenges and milestone were achieved, as we extracted and learned different resources and limitations that the PointNet architecutre offers and lacks. 

We achieved a high level experimenting with 3d data, as all predicted results and different stages were acquired visually in terms of understanding our model and its performance better. 

We also successfully deplopyed on GoogleCloud the pre-processing and training processes. 

## Future work

Further work is directly pointed into implementing PointNet++ with this dataset (with/without augmentation methods) and check where the cieling lies. We are pretty sure that our model performs good for global characteristics, as we get a pretty good glance of the city structure, building, vegetation, poles... But it is perfoming poorly on local structures such as correct size and position of cars, trucks, and some structures are not meant to be in terms of labeling but indeed in location and form.  [PointNet++](https://github.com/charlesq34/pointnet2)

By applying PointNet++ onto DALES, a hierarchical neural network that applies PointNet recursively on a nested partitioning, we could achieve greater results onto local structures of the semantic segmentation task. 

We also wondered about which balancing/augmenting methods to use, and we are aware that there is another subsampling method, called voxel subsampling, which performes a uniform subsampling only collecting the average of points for every quadrant of the grid. it could be made using open3d library onto the pointcloud DALES dataset, but for a time perspective we didn't implement this solution, as well as the computational resources that it would recquire, specialy on this concrete dataset. [Open3D library](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)

We are also aware of new State of the Art architectures that could strongly perfom onto the DALES dataset; also in terms of efficiency and number of parameters. One of them is the Point-Voxel CNN which is considered an architecture for Efficient 3D Deep Learning, that speedsup and performs better than lots of previously released models, published mostly by MIT researchers. [Point-Voxel CNN](https://arxiv.org/pdf/1907.03739.pdf)  



## Authors

- [laurahomet](https://github.com/laurahomet) - Laura Homet
- [albertpedra45](https://github.com/albertpedra45) - Albert Pedraza
- [paula22on](https://github.com/paula22on) - Paula Osés
- [dom27d](https://github.com/dom27d) - Daniel Ochavo
