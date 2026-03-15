import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from dataset import MultiSubjectNeuroFlowDataset
from model import NeuroFlexMoE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 INITIALIZING FLEX-MOE PIPELINE ON: {device}")

    raw_data_dir = "../raw_data" 
    suds_scores = [25, 25, 35, 30, 25]
    boundaries = [0, 690, 1376, 2500, 3668, 4306] 
    
    print("🔄 Loading and Slicing Dataset...")
    full_ds = MultiSubjectNeuroFlowDataset(raw_data_dir, suds_scores, boundaries)
    
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    
    model = NeuroFlexMoE(num_experts=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() 
    gate_loss_weight = 0.01 # Balances task accuracy with expert routing
    
    epochs = 15
    print(f"\n" + "="*40 + f"\n🧠 TRAINING ON {len(full_ds)} DATA WINDOWS\n" + "="*40)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_gate_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
            task_loss = criterion(preds, y)
            gate_loss = model.gate_loss()
            
            # Combine the errors!
            loss = task_loss + (gate_loss_weight * gate_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += task_loss.item()
            total_gate_loss += float(gate_loss)
            
        avg_mse = total_loss/len(train_loader)
        avg_gate = total_gate_loss/len(train_loader)
        print(f"   Epoch {epoch+1}/{epochs} | Average MSE: {avg_mse:.4f} | Gate Loss: {avg_gate:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/flexmoe_final.pth")
    print("\n💾 Training complete. Weights saved to weights/flexmoe_final.pth")

if __name__ == "__main__":
    main()
