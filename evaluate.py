import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from models.mtl_network import SoilMTLNetwork
from data.dataset import get_dummy_data, SoilDataset
from torch.utils.data import DataLoader

def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_y_m_true = []
    all_y_m_pred = []
    
    all_y_t_true = []
    all_y_t_pred = []
    
    with torch.no_grad():
        for x, (y_m, y_t) in dataloader:
            x = x.to(device)
            y_m = y_m.to(device)
            y_t = y_t.to(device)
            
            pred_m, pred_t = model(x)
            
            # Store Moisture
            all_y_m_true.extend(y_m.cpu().numpy().flatten())
            all_y_m_pred.extend(pred_m.cpu().numpy().flatten())
            
            # Store Texture
            all_y_t_true.extend(y_t.cpu().numpy())
            # Convert logits to predicted class
            pred_classes = torch.argmax(pred_t, dim=1)
            all_y_t_pred.extend(pred_classes.cpu().numpy())
            
    # Calculate Metrics
    rmse = mean_squared_error(all_y_m_true, all_y_m_pred, squared=False)
    r2 = r2_score(all_y_m_true, all_y_m_pred)
    
    accuracy = accuracy_score(all_y_t_true, all_y_t_pred)
    f1 = f1_score(all_y_t_true, all_y_t_pred, average='macro')
    
    print("=== Evaluation Results ===")
    print(f"Moisture (Regression)  -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
    print(f"Texture  (Classifier)  -> Accuracy: {accuracy:.4f}, Macro-F1: {f1:.4f}")

if __name__ == '__main__':
    print("Testing evaluation logic with a randomly initialized model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dummy Data
    in_channels, num_classes, mode = 10, 4, '1d'
    f, m, t = get_dummy_data(50, mode=mode, in_channels=in_channels, num_classes=num_classes)
    dataset = SoilDataset(f, m, t, mode=mode)
    dataloader = DataLoader(dataset, batch_size=8)
    
    # 2. Model
    model = SoilMTLNetwork(input_mode=mode, in_channels=in_channels, num_texture_classes=num_classes)
    model.to(device)
    
    evaluate_model(model, dataloader, device)
