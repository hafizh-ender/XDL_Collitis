import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.25):
        """
        Initialize ResNet50 model with custom number of classes

        Args:
            num_classes (int): Number of output classes for classification
        """
        super(ResNet50, self).__init__()

        self.resnet_model = models.resnet50(weights='IMAGENET1K_V2')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(204800, num_classes)

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

        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)

        x = self.resnet_model.layer1(x)
        x = self.resnet_model.layer2(x)
        x = self.resnet_model.layer3(x)
        features = self.resnet_model.layer4(x)
        
        # out = F.relu(features, inplace=True)
        
        # Add dropout layer
        if self.training:
            features = self.dropout(features)

        # Flatten the output
        out = torch.flatten(features, 1)

        # Pass through the classifier
        out = self.classifier(out) 
        
        return out