import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet121(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.25):
        """
        Initialize DenseNet121 model with custom number of classes
        
        Args:
            num_classes (int): Number of output classes for classification
        """
        super(DenseNet121, self).__init__()
        
        self.model = models.densenet121(weights='IMAGENET1K_V1')
        
        num_features = self.model.classifier.in_features
        
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x) 