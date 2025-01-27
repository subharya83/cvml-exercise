import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class SubjectMapGenerator(nn.Module):
    def __init__(self):
        super(SubjectMapGenerator, self).__init__()
        # Light-weight 6-layer CNN for subject map generation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.sigmoid(self.conv6(x))
        return x

class VarianceMapModule(nn.Module):
    def __init__(self, num_clips=8):
        super(VarianceMapModule, self).__init__()
        self.num_clips = num_clips

    def forward(self, feature_maps):
        # feature_maps: [N, C, H, W]
        variance_maps = []
        for i in range(self.num_clips):
            for j in range(self.num_clips):
                if i != j:
                    sim = F.cosine_similarity(feature_maps[i], feature_maps[j], dim=1)
                    variance_maps.append(sim)
        variance_map = torch.stack(variance_maps, dim=0).mean(dim=0)
        return variance_map

class SGNet(nn.Module):
    def __init__(self, num_classes_scale=5, num_classes_movement=4):
        super(SGNet, self).__init__()
        self.subject_map_generator = SubjectMapGenerator()
        self.variance_map_module = VarianceMapModule()
        self.backbone = resnet50(pretrained=True)
        self.fc_scale = nn.Linear(2048, num_classes_scale)
        self.fc_movement = nn.Linear(2048, num_classes_movement)

    def forward(self, x):
        # Generate subject map
        subject_map = self.subject_map_generator(x)
        subject_image = x * subject_map

        # Extract features from the whole image and subject image
        whole_features = self.backbone(x)
        subject_features = self.backbone(subject_image)

        # Fuse features for scale classification
        scale_features = torch.cat([whole_features, subject_features], dim=1)
        scale_logits = self.fc_scale(scale_features)

        # Generate variance map for movement classification
        variance_map = self.variance_map_module(whole_features)
        movement_features = torch.cat([whole_features, variance_map], dim=1)
        movement_logits = self.fc_movement(movement_features)

        return scale_logits, movement_logits