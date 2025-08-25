import torch
import torch.nn as nn
from src.densenet import DenseNet121
from typing import List

class DenseNet121Multilayer(nn.Module):
    def __init__(self, model_layer_1: DenseNet121, model_layer_2: List[DenseNet121]):
        """
        Initialize DenseNet121Multilayer model with custom number of classes
        
        Args:
            model_layer_1: Densenet model for classifying the group first.
            model_layer_2: list of Densenet model for classifying into binary class. pastiin index class pada model_layer_1 is consistent with the appropriate model in the list.
                            Example: model_layer_1 -> [chron_tb] and [uc_infeksi], then model_layer_2 = [chron_tb_model, uc_infeksi_model] 
        """
        super(DenseNet121Multilayer, self).__init__()
        
        self.model_layer_1 = model_layer_1
        self.model_layer_2 = model_layer_2
        
    def forward(self, x):
        """
        Forward pass of the model fro first layer to the second layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
                            or (channels, height, width) for single image
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Add batch dimension if input is a single image
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # First layer classification
        with torch.no_grad():
            predicted_group_indices = torch.argmax(self.model_layer_1(x), dim=1)
            out = self.model_layer_2[predicted_group_indices](x)
        
        return out