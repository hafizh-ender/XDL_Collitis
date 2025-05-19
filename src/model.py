import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Squash(nn.Module):
    def __init__(self, eps=1e-20):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        coef = 1 - 1 / (torch.exp(norm) + self.eps)
        unit = x / (norm + self.eps)
        return coef * unit


class PrimaryCaps(nn.Module):
    def __init__(
        self,
        in_channels, #F
        kernel_size, #K
        capsule_size, #N & D
        stride=1, #s
    ):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_capsules, self.dim_capsules = capsule_size
        self.stride = stride

        self.dw_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_capsules * self.dim_capsules,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.squash = Squash()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.dw_conv2d(x)
        
        # Calculate spatial dimensions
        spatial_size = x.size(2) * x.size(3)
        # Reshape to [batch_size, num_capsules, dim_capsules]
        x = x.view(batch_size, self.num_capsules, -1)
        return self.squash(x)


class RoutingCaps(nn.Module):
    def __init__(self, in_capsules, out_capsules):
        super(RoutingCaps, self).__init__()
        self.N0, self.D0 = in_capsules
        self.N1, self.D1 = out_capsules
        self.squash = Squash()

        # initialize routing parameters
        self.W = nn.Parameter(torch.Tensor(self.N1, self.N0, self.D0, self.D1))
        nn.init.kaiming_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(self.N1, self.N0, 1))

    def forward(self, x):
        batch_size = x.size(0)
        ## prediction vectors
        # ji,kjiz->kjz = k and z broadcast, then ji,ji->j = sum(a*b,axis=1)
        u = torch.einsum("...ji,kjiz->...kjz", x, self.W)  # (batch_size/B, N1, N0, D1)

        ## coupling coefficients
        # ij,kj->i = ij,kj->k = sum(matmul(a,a.T),axis=0) != ij,ij->i
        c = torch.einsum("...ij,...kj->...i", u, u)  # (B, N1, N0)
        c = c[..., None]  # (B, N1, N0, 1) for bias broadcasting
        c = c / torch.sqrt(torch.tensor(self.D1).float())  # stabilize
        c = torch.softmax(c, dim=1) + self.b

        ## new capsules
        s = torch.sum(u * c, dim=-2)  # (batch_size, N1, D1)
        return self.squash(s)


class CapsLen(nn.Module):
    def __init__(self, eps=1e-7):
        super(CapsLen, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.sqrt(
            torch.sum(x**2, dim=-1) + self.eps
        )  # (batch_size, num_capsules)


class CapsMask(nn.Module):
    def __init__(self):
        super(CapsMask, self).__init__()

    def forward(self, x, y_true=None):
        if y_true is not None:  # training mode
            mask = y_true
        else:  # testing mode
            # convert list of maximum value's indices to one-hot tensor
            temp = torch.sqrt(torch.sum(x**2, dim=-1))
            mask = F.one_hot(torch.argmax(temp, dim=1), num_classes=temp.shape[1])
        
        masked = x * mask.unsqueeze(-1)
        return masked.view(x.shape[0], -1)  # reshape


class EfficientCapsNet(nn.Module):
    def __init__(self, input_size):
        super(EfficientCapsNet, self).__init__()
        # input_size: (C, H, W)
        # Calculate spatial dimension after conv layers
        h = input_size[1] # Assuming H=W
        h = h - 5 + 1  # conv1 (kernel_size=5, padding=0)
        h = h - 3 + 1  # conv2 (kernel_size=3, padding=0)
        h = h - 3 + 1  # conv3 (kernel_size=3, padding=0)
        h = math.floor((h - 3) / 2 + 1) # conv4 (kernel_size=3, stride=2, padding=0)
        
        feature_map_dim_after_conv4 = h

        self.conv1 = nn.Conv2d(
            in_channels=input_size[0], out_channels=32, kernel_size=5, padding=0
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3)  # padding=0 is default
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.primary_caps = PrimaryCaps(
            in_channels=128, 
            kernel_size=feature_map_dim_after_conv4, # This makes dw_conv output 1x1 spatially
            capsule_size=(128, 64), 
            stride=1 # Stride must be 1 for 1x1 output with kernel_size = input_size
        )
        self.routing_caps = RoutingCaps(in_capsules=(128, 64), out_capsules=(3, 16))
        self.len_final_caps = CapsLen()
        self.mask = CapsMask()
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Kaiming normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x, y_true=None, mode="train"):
        def print_stats(tensor, name, is_param=False):
            if tensor is None:
                print(f"{name} is None")
                return
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"!!! NaN or Inf detected in {name} !!!")
                print(tensor)
            else:
                print(f"{name} - min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}, shape: {tensor.shape}")

        print_stats(x, "Initial input x")

        # --- Detailed check for conv1 and bn1 ---
        print("--- Checking conv1 and bn1 parameters ---")
        print_stats(self.conv1.weight, "conv1.weight", is_param=True)
        if self.conv1.bias is not None:
            print_stats(self.conv1.bias, "conv1.bias", is_param=True)
        print_stats(self.bn1.weight, "bn1.weight", is_param=True)
        print_stats(self.bn1.bias, "bn1.bias", is_param=True)
        if self.bn1.running_mean is not None:
             print_stats(self.bn1.running_mean, "bn1.running_mean", is_param=True)
        if self.bn1.running_var is not None:
             print_stats(self.bn1.running_var, "bn1.running_var", is_param=True)
        print("--- End of conv1 and bn1 parameters ---")

        out_conv1 = self.conv1(x)
        print_stats(out_conv1, "x after conv1")
        
        out_bn1 = self.bn1(out_conv1)
        print_stats(out_bn1, "x after bn1 (before relu)")
        
        x = torch.relu(out_bn1)
        print_stats(x, "x after conv1 + bn1 + relu")
        # --- End of detailed check ---
        
        x = torch.relu(self.bn2(self.conv2(x)))
        print_stats(x, "x after conv2 + bn2 + relu")
        x = torch.relu(self.bn3(self.conv3(x)))
        print_stats(x, "x after conv3 + bn3 + relu")
        x = torch.relu(self.bn4(self.conv4(x)))
        print_stats(x, "x after conv4 + bn4 + relu (input to primary_caps)")
        
        x = self.primary_caps(x)
        print_stats(x, "x after primary_caps")
        
        x = self.routing_caps(x)
        print_stats(x, "x after routing_caps (input to len_final_caps)")
        
        y_predict = self.len_final_caps(x)
        print_stats(y_predict, "y_predict (final model output)")
        
        return y_predict