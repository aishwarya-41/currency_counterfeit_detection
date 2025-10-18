import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    This module learns to weight the importance of each channel.
    """
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        avg = F.adaptive_avg_pool2d(x, (1, 1))
        avg_out = self.shared_mlp(avg)
        
        # Global max pooling
        max_ = F.adaptive_max_pool2d(x, (1, 1))
        max_out = self.shared_mlp(max_)
        
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    This module learns to focus on important spatial regions.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Use a 7x7 conv as many CBAM papers do
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling along channel
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max pooling along channel
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(cat))
        return x * scale

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Combines both Channel and Spatial attention.
    """
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

