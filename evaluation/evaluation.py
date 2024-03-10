import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.getcwd(),'../'))

from dataset import MyDataset
from model import ClassificationPointNet, SegmentationPointNet
from utils import (
    calculate_iou,
    compute_accuracy,
    compute_confusion_matrix,
    plot_accuracies,
    plot_confusion_matrix,
    plot_IoU,
    plot_iou_per_class,
    plot_losses,
    plot_iou_per_class_final,
    visualize_sample,
    visualize_100_concatenated_samples,
    write_sample_to_csv
)

def main(args):

    SEGMENTATION = True
    WEIGHTED_LOSS = False
    NUM_POINTS = 4096
    NUM_CLASSES = 9
    NUM_TEST_TILES = 11

    if args.weighted:
        WEIGHTED_LOSS = True

    train_dataset = MyDataset("../data", NUM_POINTS, "train")
    test_dataset = MyDataset("../data", NUM_POINTS, "test")

    print("Dataset ready")

    if WEIGHTED_LOSS:
        class_weights = (
            train_dataset.calculate_class_weights()
        )
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float
        )
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = torch.nn.NLLLoss()

        
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Dataloader ready")

    if SEGMENTATION:
        model = SegmentationPointNet(num_classes=NUM_CLASSES, point_dimension=3)
    else:
        model = ClassificationPointNet(
            num_classes=16, point_dimension=3, segmentation=False
        )

    if torch.cuda.is_available():
        model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    if args.naive:
        model_checkpoint = 'checkpoints-segmentation/segmentation_checkpoint_naive.pth'
    elif args.augmentation:
        model_checkpoint = 'checkpoints-segmentation/segmentation_checkpoint_augmentation.pth'
    elif args.rotated:
        model_checkpoint = 'checkpoints-segmentation/segmentation_checkpoint_rotated.pth'
    elif args.weighted:
        model_checkpoint = 'checkpoints-segmentation/segmentation_checkpoint_weighted.pth'

    if model_checkpoint:
        state = torch.load(model_checkpoint, map_location=torch.device(device))
        model.load_state_dict(state['model'])

    print(f"{model_checkpoint} loaded")

    # Testing our model
    model.eval()
    epoch_test_loss = []
    epoch_test_acc = []
    epoch_test_iou = []
    total_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    epoch_test_iou_per_class = {i: [] for i in range(NUM_CLASSES)}

    if not os.path.exists('figures/sample_images'):
        os.makedirs('figures/sample_images')
    if not os.path.exists('figures/sample_data'):
        os.makedirs('figures/sample_data')

    print("Ready to evaluate")

    with torch.no_grad():
        idx = 0
        for points, labels in test_dataloader:

            print(f"Iterating over samples, current sample idx is {idx}")

            points, labels = points.to(device), labels.to(device)
            pred, _ = model(points)

            #--- Compute loss, accuracy, iou, and confusion matrix
            print("Computing loss, accuracy, iou, and confusion matrix parameters")

            loss = criterion(pred, labels)
            epoch_test_loss.append(loss.item())

            acc = compute_accuracy(pred, labels)
            epoch_test_acc.append(acc)

            ious, mean_iou = calculate_iou(pred, labels, NUM_CLASSES)
            epoch_test_iou.append(mean_iou)
            for cls in range(NUM_CLASSES):
                epoch_test_iou_per_class[cls].append(ious[cls])

            batch_confusion_matrix = compute_confusion_matrix(
                total_confusion_matrix, pred, labels, NUM_CLASSES
            )
            total_confusion_matrix += batch_confusion_matrix

            #--- Unroll batch to visualize and store all predicted samples
            print("Creating figures of predicted samples")

            max_pred = torch.argmax(pred, dim=1)

            for batch_idx in range(points.size(0)):
                points_batch = points[batch_idx]
                pred_batch = max_pred[batch_idx]

                # Visualize and save original sample

                save_to_file = f'figures/sample_images/original_sample_{NUM_POINTS}points_{idx}.png'
                visualize_sample(points_batch, labels[batch_idx], save_to_file, "Testing")

                save_to_file = f'figures/sample_data/original_sample_{NUM_POINTS}points_{idx}.csv'
                labels_reshaped = labels[batch_idx].view(-1, 1)
                orig_sample = torch.cat((points_batch, labels_reshaped), dim=1)
                write_sample_to_csv(save_to_file, orig_sample)

                # Visualize and save predicted sample

                save_to_file = f'figures/sample_images/predicted_sample_{NUM_POINTS}points_{idx}.png'
                visualize_sample(points_batch, pred_batch, save_to_file, "Testing")

                save_to_file = f'figures/sample_data/predicted_sample_{NUM_POINTS}points_{idx}.csv'
                pred_reshaped = pred_batch.view(-1, 1)
                pred_sample = torch.cat((points_batch, pred_reshaped), dim=1)
                write_sample_to_csv(save_to_file, pred_sample)

                idx += 1


    #--- After completing the loop over the test_dataloader
                
    average_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
    average_test_accuracy = sum(epoch_test_acc) / len(epoch_test_acc)
    average_test_iou = sum(epoch_test_iou) / len(epoch_test_iou)
    avg_test_iou_per_class = []
    for cls in range(NUM_CLASSES):
                avg_test_iou_per_class.append(np.mean(epoch_test_iou_per_class[cls]))

    # Print the results
    print(
        f"Test Results - Loss: {average_test_loss:.4f}, Accuracy: {average_test_accuracy:.2f}, IoU: {average_test_iou:.2f}%"
    )

    # Plotting the results
    output_folder = "figures"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    class_names = [
        "Others",
        "Ground",
        "Vegetation",
        "Cars",
        "Trucks",
        "Powerlines",
        "Fences",
        "Poles",
        "Buildings"
    ]

    # testing
    if args.naive:
        model_config = 'naive'
    elif args.augmentation:
        model_config = 'augmentation'
    elif args.rotated:
        model_config = 'rotated'
    elif args.weighted:
        model_config = 'weighted'

    plot_iou_per_class_final(
        avg_test_iou_per_class,
        class_names,
        phase="Testing",
        save_to_file=os.path.join(
            output_folder, "iou_test_per_class_plot" + str(NUM_POINTS) + model_config + ".png"
        ),
    )

    plot_confusion_matrix(
        total_confusion_matrix,
        class_names,
        save_to_file=os.path.join(
            output_folder, "confmatrix_plot" + str(NUM_POINTS) + model_config + ".png"
        )
    )


    #--- Visualize and save concatenation of 100 samples to recreate original and predicted tile
                
    for idx in range(0, NUM_TEST_TILES*100, 100):
        
        print(f"Creating concatenated figures of 100 samples. Currently on idx {idx}")

        sample_data_path  = os.path.join(os.getcwd(), 'figures/sample_data')
        sample_image_path = os.path.join(os.getcwd(), 'figures/sample_images')

        visualize_100_concatenated_samples(
            sample_data_path  = sample_data_path, 
            sample_image_path = sample_image_path, 
            filename_no_idx = f'original_sample_{NUM_POINTS}points',
            start_idx = idx
        )

        visualize_100_concatenated_samples(
            sample_data_path  = sample_data_path, 
            sample_image_path = sample_image_path, 
            filename_no_idx   = f'predicted_sample_{NUM_POINTS}points',
            start_idx = idx
        )


#--- Main

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation script")

    parser.add_argument(
        "--naive", action="store_true", help="Evaluate naive approach"
    )
    parser.add_argument(
        "--augmentation", action="store_true", help="Evaluate naive approach"
    )
    parser.add_argument(
        "--rotated", action="store_true", help="Evaluate naive approach"
    )
    parser.add_argument(
        "--weighted", action="store_true", help="Evaluate naive approach"
    )
    args = parser.parse_args()

    main(args)