import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision.models as models

class MobileNetV3(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.25):
        """
        Initialize MobileNetV3 model with custom number of classes

        Args:
            num_classes (int): Number of output classes for classification
        """
        super(MobileNetV3, self).__init__()

        self.mobilenet_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(57600, num_classes)

    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
                            or (channels, height, width) for single image
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Add batch dimension if input is a single image
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.mobilenet_model.features(x)
        
        if self.training:
            x = self.dropout(x)
        
        # x = self.mobilenet_model.avgpool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)