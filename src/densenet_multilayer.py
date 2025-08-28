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
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # First layer classification
        self.out_1 = self.model_layer_1(x)
        pred_idx = torch.argmax(self.out_1, dim=1).item()

        if pred_idx == 0:
            out_2_1 = self.model_layer_2[pred_idx](x) 
            out_2_2 = torch.zeros_like(out_2_1, device=out_2_1.device, dtype=out_2_1.dtype)
        elif pred_idx == 1:
            out_2_2 = self.model_layer_2[pred_idx](x) 
            out_2_1 = torch.zeros_like(out_2_2, device=out_2_2.device, dtype=out_2_2.dtype)

        # Concatenate and return 1D tensor (for B == 1)
        out_2 = torch.cat([out_2_1, out_2_2], dim=-1).view(-1)
        return out_2