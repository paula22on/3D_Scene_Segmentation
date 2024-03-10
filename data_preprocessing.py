import argparse
import os

import laspy
import numpy as np

from utils import balance_classes, sample_random_rotation_z_axis, write_sample_to_csv


def exportSubsamples(idx, sample_type, divider, subsamples, balance, rotate):
    """
    Exports the processed subsamples into CSV files within a specified directory structure.

    Parameters:
        idx (int): Index to help generate unique file names.
        sample_type (str): Indicates whether the samples are for 'train' or 'test' data.
        divider (int): The number used to divide the original sample into smaller regions.
        subsamples (list): The list of subsampled point cloud data.
        balance (bool): Boolean to apply balance the data.
        rotate (bool): Boolean to apply rotation the data to do data augmentation.
    """
    # Prepare output directory based on sample type (train or test).
    outdir = f"data/{sample_type}"
    print(f"Exporting subsamples into {outdir} directory")

    # Create the directory if it doesn't exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Directory '{outdir}' created successfully.")

    # Calculate the starting index for subsamples, adjusted for training data.
    nsub = idx * divider**2
    if sample_type == "train" and balance and rotate:
        nsub = nsub * 2

    # Iterate over the subsamples grid to process and save each subsample.
    for i in range(len(subsamples)):
        for j in range(len(subsamples)):
            sample = np.array(subsamples[i][j])

            if len(sample) > 0:  # Check if the subsample contains points.
                original_sample_saved = False

                if sample_type == "train":
                    if balance:
                        sample = balance_classes(
                            sample
                        )  # Balance the classes in the subsample.
                        write_sample_to_csv(
                            f"{outdir}/{divider}_divisions_{nsub}.csv", sample
                        )
                        nsub += 1
                    if rotate:
                        rotated_sample = sample_random_rotation_z_axis(
                            sample
                        )  # Apply a random rotation for data augmentation and save the rotated sample.
                        write_sample_to_csv(
                            f"{outdir}/{divider}_divisions_{nsub}.csv", rotated_sample
                        )
                        nsub += 1
                    # If the sample was not balanced and not rotated, just save the original sample.
                    if not original_sample_saved and not rotate:
                        write_sample_to_csv(
                            f"{outdir}/{divider}_divisions_{nsub}.csv", sample
                        )
                        nsub += 1

                else:  # For testing samples, just save without balancing or rotation.
                    write_sample_to_csv(
                        f"{outdir}/{divider}_divisions_{nsub}.csv", sample
                    )
                    nsub += 1

            


def subsample(divider, X, Y, Z, labels):
    """
    Subdivides the point cloud data into smaller regions based on the divider parameter.

    Parameters:
        divider (int): The number of divisions along each axis.
        X, Y, Z (array-like): Coordinates of points in the point cloud.
        labels (array-like): The classification labels for each point.

    Returns:
        list: A nested list of subdivided point cloud regions.
    """
    # Initialize a grid to hold subsamples.
    alldivisions = []
    for i in range(divider):
        row = []
        for j in range(divider):
            row.append([])
        alldivisions.append(row)

    # Calculate the size of each subsample division.
    areaX = max(X) // divider
    areaY = max(Y) // divider

    # Assign each point to a subsample based on its coordinates.
    for i in range(len(labels)):
        xidx = min(X[i] // areaX, divider - 1)
        yidx = min(Y[i] // areaY, divider - 1)
        alldivisions[xidx][yidx].append((X[i], Y[i], Z[i], labels[i]))

        if (i + 1) % (len(labels) // 100) == 0:
            print(f"Subsampling, {int((i + 1) / len(labels) * 100)}% processed")

    return alldivisions


def normalize(las_x, las_y):
    """
    Normalizes the X and Y coordinates of the point cloud data.

    Parameters:
        las_x, las_y (array-like): The X and Y coordinates of the points.

    Returns:
        tuple: Normalized X and Y coordinates.
    """
    # Normalize X and Y coordinates by subtracting the minimum value.
    return las_x - min(las_x), las_y - min(las_y)


def simplify(las):
    """
    Simplifies the LAS file data by extracting coordinates and classification labels.

    Parameters:
        las (laspy.file.File): The LAS file object.

    Returns:
        tuple: X, Y, Z coordinates and classification labels of the point cloud.
    """
    # Extract relevant information from the .las file.
    X = las.X
    Y = las.Y
    Z = las.Z
    classification = las.classification

    # Prepare the labels from classification.
    labels = []
    for i in range(len(las.X)):
        labels.append(classification[i])

        if (i + 1) % (len(las.X) // 100) == 0:
            print(f"Reading labels, {int((i + 1) / len(las.X) * 100)}% processed")

    return X, Y, Z, labels


def process(path, sample, idx, sample_type, divider, balance, rotate):
    """
    Main processing function to read a LAS file, normalize and subsample the data,
    and export it into CSV format for further use.

    Parameters:
        path (str): Directory path containing the LAS files.
        sample (str): The name of the LAS file to process.
        idx (int): Sample index for unique file naming.
        sample_type (str): 'train' or 'test' to distinguish the sample purpose.
        divider (int): The number used for subdividing the point cloud.
    """
    # Start processing of a given sample file.
    print(f"Starting processing of {sample_type} sample {sample}")

    # Read the .las file.
    las = laspy.read(f"{path}/{sample}")

    # Simplify and normalize the point cloud data.
    X, Y, Z, labels = simplify(las)
    X, Y = normalize(X, Y)

    # Subsample the data.
    subdivisions = subsample(divider, X, Y, Z, labels)

    # Export the subsampled data.
    exportSubsamples(idx, sample_type, divider, subdivisions, balance, rotate)


if __name__ == "__main__":
    # Setup the argument parser and read arguments
    parser = argparse.ArgumentParser(description="Data pre-processing script")
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to the directory containing DALES dataset in .las format",
    )
    parser.add_argument("divider", type=int, help="Divider used for subsampling")
    parser.add_argument(
        "--balance", action="store_true", help="Balance the classes in the dataset"
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Apply random rotation around Z-axis for training data",
    )

    args = parser.parse_args()

    train_path = f"{args.dir_path}/train"
    test_path = f"{args.dir_path}/test"

    train_samples = os.listdir(train_path)
    test_samples = os.listdir(test_path)

    # Process each LAS file in the train and test directories
    for i, sample in enumerate(train_samples):
        process(train_path, sample, i, "train", args.divider, args.balance, args.rotate)

    for i, sample in enumerate(test_samples):
        process(test_path, sample, i, "test", args.divider, args.balance, args.rotate)
