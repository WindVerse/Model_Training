import torch
import torch.nn as nn

import config as cfg

class L2Loss(nn.Module):
    def __init__(self, mean, std, device="cuda"):
        super().__init__()
        # Shape: [N]
        self.pin_mask = cfg.PIN_MASK.to(device).squeeze()
        # Move stats to device so they don't cause crashes during math
        self.target_mean = mean.to(device)
        self.target_std = std.to(device)
    
    def forward(self, pred_raw, target_raw, curr_pos=None, curr_vel=None):
        # pred_raw and target_raw shape: [B*T, N, 3]
        
        # 1. Normalize ONLY the target (if not done in dataset)
        if cfg.MODEL == 'MeshGraphNet':   
            target_norm = (target_raw - self.target_mean) / (self.target_std + 1e-8)
        else:
            target_norm = target_raw
            
        # The network natively outputs in the normalized space
        pred_norm = pred_raw
        
        # 2. Calculate squared error
        # Shape becomes [B*T, N]
        error = torch.sum((pred_norm - target_norm) ** 2, dim=-1)
        
        # 3. Create a boolean mask for FREE nodes (shape: [N])
        free_node_mask = (self.pin_mask == 0.0)
        
        # 4. Apply mask using PyTorch Broadcasting!
        # This selects all batches/frames, but strictly slices out the free nodes
        masked_error = error[:, free_node_mask]
        
        # 5. Calculate the mean loss over only the free nodes
        loss = torch.mean(masked_error)
        
        # Create a dummy zero tensor on the same device so .item() doesn't crash in the loop
        z = torch.tensor(0.0, device=pred_raw.device)
        
        # Return loss for RMSE as well (or calculate actual RMSE if you prefer), 
        # and zeros for the unused physics constraints
        return loss, loss, z, z, z, z, z, z