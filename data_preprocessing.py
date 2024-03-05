import argparse
import os
import laspy
import numpy as np
from collections import Counter

def random_rotation_z_axis(points, theta=None):
    #print(f"Points are: {points}")
    if theta is None:
        theta = np.random.uniform(0, 360)  # Random rotation angle
    #print(f"Angle is {theta} deg")
    cos_val = np.cos(np.radians(theta))
    sin_val = np.sin(np.radians(theta))
    rotation_matrix = np.array([[cos_val, -sin_val, 0],
                                [sin_val, cos_val, 0],
                                [0, 0, 1]])  # Rotation matrix for Z-axis
    rotated_points = np.dot(points[:, :3], rotation_matrix)  # Apply rotation
    #print(f"Rotated points are: {rotated_points}")
    return np.hstack((rotated_points, points[:, 3:]))  # Reattach labels

def balance_classes(sample):
    labels = sample[:, -1]
    label_distribution = Counter(labels)
    average_num_points = int(np.mean(list(label_distribution.values())))
    balanced_sample = []
    for label, count in label_distribution.items():
        label_samples = sample[labels == label]
        if count > average_num_points:
            indices = np.random.choice(len(label_samples), size=average_num_points, replace=False)
        else:
            indices = np.random.choice(len(label_samples), size=average_num_points-count, replace=True)
            balanced_sample.extend(label_samples) # Keep points that were already present
        balanced_sample.extend(label_samples[indices])
    return np.array(balanced_sample)

def exportSubsamples(idx, sample_type, divider, subsamples, theta):
    outdir = f"data/{sample_type}"

    print(f"Exporting subsamples into {outdir} directory")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Directory '{outdir}' created successfully.")

    nsub = idx * divider**2
    if sample_type == "train": nsub = nsub * 2
    for i in range(len(subsamples)):
        for j in range(len(subsamples)):
            sample = np.array(subsamples[i][j])
            if len(sample) > 0:  # Check if sample is not empty
                if sample_type == "train":
                    sample = balance_classes(sample)
                    with open(f"{outdir}/{divider}_divisions_{nsub}.csv", "w") as csv_file:
                        for item in sample:
                            X, Y, Z, label = item
                            csv_file.write(f"{X},{Y},{Z},{label}\n")
                    rotated_sample = random_rotation_z_axis(sample)  # Apply the same rotation to all points
                    nsub += 1
                    with open(f"{outdir}/{divider}_divisions_{nsub}.csv", "w") as csv_file:
                        for item in rotated_sample:
                            X, Y, Z, label = item
                            csv_file.write(f"{X},{Y},{Z},{label}\n")
                else: #sample_type == test
                    with open(f"{outdir}/{divider}_divisions_{nsub}.csv", "w") as csv_file:
                        for item in sample:
                            X, Y, Z, label = item
                            csv_file.write(f"{X},{Y},{Z},{label}\n")
            nsub += 1


def subsample(divider, X, Y, Z, labels):
    alldivisions = []
    for i in range(divider):
        row = []
        for j in range(divider):
            row.append([])
        alldivisions.append(row)

    areaX = max(X) // divider
    areaY = max(Y) // divider

    for i in range(len(labels)):
        xidx = min(X[i] // areaX, divider - 1)
        yidx = min(Y[i] // areaY, divider - 1)
        alldivisions[xidx][yidx].append((X[i], Y[i], Z[i], labels[i]))

        if (i + 1) % (len(labels) // 100) == 0:
            print(f"Subsampling, {int((i + 1) / len(labels) * 100)}% processed")

    return alldivisions


def normalize(las_x, las_y):
    return las_x - min(las_x), las_y - min(las_y)


def simplify(las):
    X = las.X
    Y = las.Y
    Z = las.Z
    classification = las.classification

    labels = []
    for i in range(len(las.X)):
        labels.append(classification[i])

        if (i + 1) % (len(las.X) // 100) == 0:
            print(f"Reading labels, {int((i + 1) / len(las.X) * 100)}% processed")

    return X, Y, Z, labels


def process(path, sample, idx, sample_type, divider):
    print(f"Starting processing of {sample_type} sample {sample}")
    las = laspy.read(f"{path}/{sample}")
    X, Y, Z, labels = simplify(las)
    X, Y = normalize(X, Y)
    subdivisions = subsample(divider, X, Y, Z, labels)
    exportSubsamples(idx, sample_type, divider, subdivisions, theta=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to the directory containing DALES dataset in .las format",
    )
    parser.add_argument("divider", type=int, help="Divider used for subsampling")
    args = parser.parse_args()

    train_path = f"{args.dir_path}/train"
    test_path = f"{args.dir_path}/test"

    train_samples = os.listdir(train_path)
    test_samples = os.listdir(test_path)

    for i, sample in enumerate(train_samples):
        process(train_path, sample, i, "train", args.divider)

    for i, sample in enumerate(test_samples):
        process(test_path, sample, i, "test", args.divider)
