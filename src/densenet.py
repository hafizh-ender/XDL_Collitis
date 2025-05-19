import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision.models as models

class DenseNet121(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.25):
        """
        Initialize DenseNet121 model with custom number of classes
        
        Args:
            num_classes (int): Number of output classes for classification
        """
        super(DenseNet121, self).__init__()
        
        self.densenet_model = models.densenet121(weights='IMAGENET1K_V1')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(82944, num_classes)

        
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        features = self.densenet_model.features(x)
        out = F.relu(features, inplace=True)
        
        # Add dropout layer
        out = self.dropout(out)

        # Flatten the output
        out = torch.flatten(out, 1)

        # Pass through the classifier
        out = self.classifier(out) 
        
        return out