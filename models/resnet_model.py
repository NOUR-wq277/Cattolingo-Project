import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def build_resnet50(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model
