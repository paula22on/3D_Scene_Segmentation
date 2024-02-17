import argparse
import os

import laspy

index = 0


def exportSubsamples(idx, sample_type, divider, subsamples):
    outdir = f"data/{sample_type}"

    print(f"Exporting subsamples into {outdir} directory")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Directory '{outdir}' created successfully.")

    nsub = idx * divider**2
    for i in range(len(subsamples)):
        for j in range(len(subsamples)):
            sample = subsamples[i][j]
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
    exportSubsamples(idx, sample_type, divider, subdivisions)


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
