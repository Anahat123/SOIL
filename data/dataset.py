import torch
from torch.utils.data import Dataset
import numpy as np

class SoilDataset(Dataset):
    """
    Custom PyTorch Dataset for Soil Moisture and Texture.
    Supports returning multi-task targets: (moisture_value, texture_class_id)
    """
    def __init__(self, features, moisture_labels, texture_labels, mode='1d'):
        """
        Args:
            features (np.ndarray): Input features. 
                                   Shape (N, C) for '1d', or (N, C, H, W) for '2d'.
            moisture_labels (np.ndarray): Continuous targets for regression. Shape (N,)
            texture_labels (np.ndarray): Integer class IDs for classification. Shape (N,)
            mode (str): '1d' or '2d'.
        """
        self.features = features
        self.moisture_labels = moisture_labels
        self.texture_labels = texture_labels
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y_moisture = self.moisture_labels[idx]
        y_texture = self.texture_labels[idx]
        
        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_moisture_tensor = torch.tensor([y_moisture], dtype=torch.float32)
        y_texture_tensor = torch.tensor(y_texture, dtype=torch.long)
        
        return x_tensor, (y_moisture_tensor, y_texture_tensor)

def get_dummy_data(num_samples=100, mode='1d', in_channels=10, num_classes=4):
    """
    Generates dummy data for testing the dataloaders and model pipelines.
    """
    if mode == '1d':
        features = np.random.randn(num_samples, in_channels)
    else:
        features = np.random.randn(num_samples, in_channels, 32, 32)
        
    moisture_labels = np.random.uniform(0.05, 0.50, size=num_samples) # Volumetric water content
    texture_labels = np.random.randint(0, num_classes, size=num_samples) # e.g. Sand, Silt, Clay, Loam
    
    return features, moisture_labels, texture_labels
