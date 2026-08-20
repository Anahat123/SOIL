import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.mtl_network import SoilMTLNetwork
from data.dataset import SoilDataset, get_dummy_data
from tqdm import tqdm

def train_mtl_model(epochs=10, batch_size=16, mode='1d'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    in_channels = 10  # E.g., 10 spectral bands
    num_classes = 4   # E.g., Sand, Silt, Clay, Loam
    
    print("Generating synthetic data for testing the pipeline...")
    features, m_labels, t_labels = get_dummy_data(num_samples=200, mode=mode, in_channels=in_channels, num_classes=num_classes)
    dataset = SoilDataset(features, m_labels, t_labels, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    model = SoilMTLNetwork(input_mode=mode, in_channels=in_channels, num_texture_classes=num_classes)
    model.to(device)
    
    # 3. Setup Optimizers and Loss Functions
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Regression Loss for Moisture
    criterion_moisture = nn.MSELoss()
    # Classification Loss for Texture
    criterion_texture = nn.CrossEntropyLoss()
    
    # Loss weighting
    weight_moisture = 1.0
    weight_texture = 1.0
    
    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_m_loss = 0.0
        epoch_t_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, (y_m, y_t) in progress_bar:
            x = x.to(device)
            y_m = y_m.to(device)
            y_t = y_t.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_m, pred_t = model(x)
            
            # Compute losses
            loss_m = criterion_moisture(pred_m, y_m)
            loss_t = criterion_texture(pred_t, y_t)
            
            # Combined Loss
            total_loss = (weight_moisture * loss_m) + (weight_texture * loss_t)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Tracking
            epoch_loss += total_loss.item()
            epoch_m_loss += loss_m.item()
            epoch_t_loss += loss_t.item()
            
            progress_bar.set_postfix({'Total': total_loss.item(), 'Moisture': loss_m.item(), 'Texture': loss_t.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == '__main__':
    print("--- Testing 1D Spectral Pipeline ---")
    train_mtl_model(epochs=3, mode='1d')
    print("\n--- Testing 2D Spatial Pipeline ---")
    train_mtl_model(epochs=3, mode='2d')
