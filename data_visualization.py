import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

def visualize_points(subsample):
        X1 = []; Y1 = []; Z1 = []; L1 = []
        for x,y,z,l in subsample:
            X1.append(x)
            Y1.append(y)
            Z1.append(z)
            L1.append(l)

        X1 = np.array(X1, dtype=np.uint64)
        Y1 = np.array(Y1, dtype=np.uint64)
        Z1 = np.array(Z1, dtype=np.uint64)
        L1 = np.array(L1, dtype=np.uint8)

        print(f"Number of points {len(X1)}")

        cdict = {1: 'blue', 2: 'green', 3: 'purple', 4:'orange', 5: 'yellow', 6:'white', 7:'pink', 8:'red'}  
        fig = plt.figure(figsize=[20,20])
        ax = fig.add_subplot(111, projection='3d')

        for classification in np.unique(L1)[1:]:
            color = cdict.get(classification, 'black')  
            ax.scatter(
                 X1[L1 == classification],  
                 Y1[L1 == classification],  
                 Z1[L1 == classification],  
                 s = 25, c=color)
        # ax.view_init(90, 0)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('LAS Point Cloud Visualization')
        plt.show()  

def import_points(csv_file):
    container = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            container.append(row)
    return container

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("sample_path", type=str, help="Path to one subsample of the DALES dataset in .csv format")
    args = parser.parse_args()

    subsample = import_points(args.sample_path)
    visualize_points(subsample)



