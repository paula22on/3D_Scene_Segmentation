import torch
import torch.nn as nn
import torch.nn.functional as F


# PointNet architecture
class TransformationNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        # You can increase/decrease model capacity by adding/deleting layers and units per layer
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim * self.output_dim)

        # Adjusted batch normalization layers to match new dimensions
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

    def forward(self, x):
        num_points = x.shape[1]
        print(num_points)
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim).unsqueeze(0).to(x.device)
        identity_matrix = identity_matrix.repeat(
            x.size(0), 1, 1
        )  # Repeat for each item in the batch
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):
    def __init__(self, point_dimension, segmentation=True):
        super(BasePointNet, self).__init__()

        self.segmentation = segmentation

        self.input_transform = TransformationNet(
            input_dim=point_dimension, output_dim=point_dimension
        )
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)

        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 128, 1)
        self.conv_4 = nn.Conv1d(128, 256, 1)
        self.conv_5 = nn.Conv1d(256, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(128)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(1024)

    def forward(self, x):

        segmentation = self.segmentation

        if segmentation:
            x = x.transpose(2, 1)

        input_transform = self.input_transform(x)  # T-Net tensor [batch, 3, 3]
        x = torch.bmm(x, input_transform)  # Batch matrix-matrix product
        x = x.transpose(
            2, 1
        )  # Transpose back to [batch_size, num_features, num_points]

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))

        if segmentation:
            per_point_features = x.transpose(2, 1)
            feature_transform = self.feature_transform(
                per_point_features
            )  # T-Net tensor [batch, 64, 64]
            per_point_features = torch.bmm(per_point_features, feature_transform)
            x = per_point_features.transpose(2, 1)

        else:
            x = x.transpose(2, 1)
            feature_transform = self.feature_transform(x)
            per_point_features = 0
            x = torch.bmm(x, feature_transform)

        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x, ix = nn.MaxPool1d(x.size(-1), return_indices=True)(x)
        global_features = x.view(-1, 1024)  # Global feature vector

        return global_features, feature_transform, per_point_features, ix


class ClassificationPointNet(nn.Module):
    def __init__(self, num_classes, dropout=0.3, point_dimension=3):
        super(ClassificationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_classes)

        self.bn_1 = nn.BatchNorm1d(128)
        self.bn_2 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x, feature_transform, tnet_out, ix_maxpool = self.base_pointnet(x)

        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)

        return (
            F.log_softmax(self.fc_3(x), dim=1),
            feature_transform,
            tnet_out,
            ix_maxpool,
        )


class SegmentationPointNet(nn.Module):
    def __init__(self, point_dimension, num_classes):
        super(SegmentationPointNet, self).__init__()

        self.base_pointnet = BasePointNet(point_dimension)
        self.num_classes = num_classes

        # Layers after concatenating global features with per-point features
        self.conv_1 = nn.Conv1d(1024 + 64, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, 64, 1)
        self.conv_4 = nn.Conv1d(128, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)
        self.bn_4 = nn.BatchNorm1d(num_classes)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.transpose(
            2, 1
        )  # we transpose to [batch_size, num_features, num_points] to match BasePointNet

        global_features, feature_transform, per_point_features, _ = self.base_pointnet(
            x
        )
        # Shape of global_features: torch.Size([32, 256])
        # Shape of feature_transform: torch.Size([32, 64, 64])
        # Shape of per_point_features: torch.Size([32, 500, 64])

        # Expand global features to concatenate with per-point features
        global_features_expanded = global_features.unsqueeze(2).expand(
            -1, -1, per_point_features.shape[1]
        )
        # Shape of global_features_expanded: torch.Size([32, 256, 500])

        # Transpose per points features to match the dimensions for concatenation
        per_point_features = per_point_features.transpose(1, 2)

        # Concatenate global features with per-point features
        x = torch.cat([per_point_features, global_features_expanded], 1)

        # Additional layers for segmentation
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))

        return self.logsoftmax(x), feature_transform
