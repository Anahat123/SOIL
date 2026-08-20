import torch
import torch.nn as nn
import torchvision.models as models

class Spectral1DCNN(nn.Module):
    """
    A 1D CNN backbone designed for pixel-wise multi-spectral or hyperspectral inputs.
    """
    def __init__(self, in_channels, out_features=128):
        super(Spectral1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # Adaptive pooling to ensure a fixed size output regardless of sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        # If input is just pixel channels without sequence, sequence_length = 1
        if len(x.shape) == 2:
            x = x.unsqueeze(2)  # Add sequence dimension
            
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

class Spatial2DResNet(nn.Module):
    """
    A 2D ResNet backbone designed for spatial image patches (e.g., 32x32 tiles).
    """
    def __init__(self, in_channels, out_features=128, pretrained=False):
        super(Spatial2DResNet, self).__init__()
        # Use ResNet18 as a lightweight backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer if in_channels != 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom projection head
        self.fc = nn.Linear(resnet.fc.in_features, out_features)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
