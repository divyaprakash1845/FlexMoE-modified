import os
import glob
import torch
from torch.utils.data import Dataset

class MultiSubjectNeuroFlowDataset(Dataset):
    def __init__(self, raw_data_dir, suds_scores, boundaries, window_size=750):
        self.window_size = window_size
        self.x_data, self.y_data = [], []

        tensor_files = glob.glob(os.path.join(raw_data_dir, "**", "fused_tensor.pt"), recursive=True)
        
        if not tensor_files:
            raise FileNotFoundError("⚠️ No .pt files found! Run preprocess.py first.")

        boundary_indices = [int(b * 500) for b in boundaries]

        for tensor_path in tensor_files:
            fused_data = torch.load(tensor_path, weights_only=False)
            
            for i in range(len(suds_scores)):
                start_idx = boundary_indices[i]
                end_idx = min(boundary_indices[i+1], fused_data.shape[0])
                phase_data = fused_data[start_idx:end_idx, :]
                
                num_windows = phase_data.shape[0] // self.window_size
                for w in range(num_windows):
                    window = phase_data[w*self.window_size : (w+1)*self.window_size, :]
                    self.x_data.append(window.unsqueeze(0)) 
                    self.y_data.append(suds_scores[i])

        self.x = torch.cat(self.x_data, dim=0)
        self.y = torch.tensor(self.y_data, dtype=torch.float32)

    def __len__(self): 
        return len(self.x)
        
    def __getitem__(self, idx):
        x_val = self.x[idx]
        x_norm = (x_val - x_val.mean(0)) / (x_val.std(0) + 1e-8)
        return x_norm, self.y[idx]
