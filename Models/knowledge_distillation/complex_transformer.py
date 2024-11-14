import torch
import torch.nn as nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        
    def forward(self, x):
        residual = x  
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x + residual  

# input_tensor = torch.randn(64, 24, 6)  
class ComplexTransformer(nn.Module):
    def __init__(self):
        super(ComplexTransformer, self).__init__()
        
        self.downsample1 = DownsampleBlock(6, 12)

        self.res_block1 = ResidualBlock(12, 24)
        
        self.downsample2 = DownsampleBlock(12, 24)
        
        self.res_block2 = ResidualBlock(24, 24)
        
        self.upsample1 = UpsampleBlock(24, 12)
        
        self.res_block3 = ResidualBlock(12, 12)
        
        self.upsample2 = UpsampleBlock(12, 6)

    def forward(self, x):
        batch_size, seq_length, feature_size = x.shape
        x = x.view(batch_size * seq_length, feature_size)

        
        x = self.downsample1(x)
    
        x = self.res_block1(x)
        
        x = self.downsample2(x)

        x = self.res_block2(x)
   
        x = self.upsample1(x)
        
        x = self.res_block3(x)
        
        x = self.upsample2(x)
        
        x = x.view(batch_size, seq_length, feature_size)
        return x

