import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_generate import*
import torchvision.models as models
import random

SCENE_DICT = {
    1:"normal"
}

_BATCHNORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)

def hasBatchNorm(module: nn.Module) -> bool:
    return any(isinstance(child, _BATCHNORM_TYPES) for child in module.modules())


def disableBatchNormRunningStats(module: nn.Module) -> int:
    disabled_count = 0
    for child in module.modules():
        if isinstance(child, _BATCHNORM_TYPES):
            child.track_running_stats = False
            child.running_mean = None
            child.running_var = None
            child.num_batches_tracked = None
            disabled_count += 1
    return disabled_count


# Residual block definition
def replaceBnWithGn(model, num_groups=32):
    '''Replace BatchNorm with GropNorm
    '''
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.GroupNorm(num_groups, module.num_features))
        else:
            replaceBnWithGn(module, num_groups)

def replaceBnWithLn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, LayerNorm2d(module.num_features))
        elif isinstance(module, nn.BatchNorm1d):
            setattr(model, name, nn.LayerNorm(module.num_features))
        else:
            replaceBnWithLn(module)

    
def BuildResnet34CBAM_GroupNorm(out_dim = 512):
    base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    replaceBnWithGn(base_model, num_groups=16)
    return nn.Sequential(
        *list(base_model.children())[:-2],
        CBAM(512)
    )

def BuildResnet34CBAM_LayerNorm(outdim = 512):
    base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    replaceBnWithLn(base_model)
    return nn.Sequential(
        *list(base_model.children())[:-2],
        CBAM(512)
    )

class ResidualBlockFC(nn.Module):
    '''A residual block for full connection
    Args:
        linear:
        norm
        dropout
        activation
        projection
    '''
    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.activation = nn.ReLU()
        
        self.projection = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.activation(out)

        if self.projection is not None:
            residual = self.projection(residual)

        out = out + residual 
        return out
        
class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # (B, C, H*W)
        x = x.transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)  
        x = x.transpose(1, 2).view(B, C, H, W)
        return x
    
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_att(x)
        x_channel = x * channel_att
        
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_att(spatial_att_input)
        
        return x_channel * spatial_att

# Simple CNN for feature extraction
class BlockFeatureCNN(nn.Module):
    def __init__(self, in_channels=1, feature_dim=64):
        super(BlockFeatureCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x(H/2)x(W/2)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Output: (batch, 64, 1, 1)
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # (batch, 64)
        return self.fc(x)  # (batch, feature_dim)
    
def FeaBlockify(tensor, n_blocks, m_blocks):
    B, C, H, W = tensor.shape
    h, w = H // n_blocks, W // m_blocks
    blocks = tensor.unfold(2, h, h).unfold(3, w, w)  # shape: (B, C, n, m, h, w)
    blocks = blocks.contiguous().view(-1, C, h, w)   # (B*n*m, C, h, w)
    return blocks


def create_default_K(img_size, device):
    H, W = img_size
    K = torch.eye(3, device=device)
    K[0, 0] = K[1, 1] = 0.5 * (H + W)
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    return K.unsqueeze(0)  # (1, 3, 3)
