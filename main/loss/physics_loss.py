import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config as cfg

def compute_chamfer_loss(self, pred_pos, target_pos):
        """
        Computes symmetric Chamfer Distance between two point clouds.
        pred_pos, target_pos: (B, N, 3)
        """
        # 1. Compute pairwise distances (B, N, N)
        # cdist is an optimized C++ implementation in PyTorch
        dists = torch.cdist(pred_pos, target_pos) 
        
        # 2. For each point in pred, find closest point in target
        min_dist_pred, _ = torch.min(dists, dim=2) # (B, N)
        
        # 3. For each point in target, find closest point in pred
        min_dist_target, _ = torch.min(dists, dim=1) # (B, N)
        
        # 4. Average the distances
        return torch.mean(min_dist_pred) + torch.mean(min_dist_target)

class PhysicsLoss(nn.Module):
    def __init__(self, 
                 initial_flag_pos,  # (N, 3) 
                 mean,              # Training set Mean
                 std,               # Training set Std
                 device="cuda"):
        super().__init__()

        # 1. Hyperparameters
        self.lambda_rmse = cfg.LAMBDA_RMSE
        self.lambda_chamfer = cfg.LAMBDA_CHAMFER
        self.lambda_edge = cfg.LAMBDA_EDGE    # Penalize stretching
        self.lambda_pin = cfg.LAMBDA_PIN      # Penalize moving pinned nodes
        self.dt = cfg.DELTA_T

        # 2. Normalization Stats (For De-normalization)
        self.mean = torch.as_tensor(mean, device=device).view(1, 1, -1)
        self.std = torch.as_tensor(std, device=device).view(1, 1, -1)
        
        # We need specific indices for Accel (usually last 3 channels if mean is huge)
        # Assuming mean/std are shape (1, 1, 3) for just acceleration, 
        # OR if they are (1, 1, 9) [pos, vel, acc], we need to slice them carefully.
        # For simplicity here, I assume mean/std correspond exactly to the model output channels.

        # 3. Load Topology (Edges)
        edge_index = np.load(cfg.TOPOLOGY_PATH)
        self.src = torch.from_numpy(edge_index[0]).long().to(device)
        self.dst = torch.from_numpy(edge_index[1]).long().to(device)

        # 4. Setup Rest Lengths (The "Springs")
        initial_pos = torch.as_tensor(initial_flag_pos, device=device).float()
        pos_only = initial_pos[:, :3] 
        
        # Calculate initial distances (L0)
        rest_vec = pos_only[self.src] - pos_only[self.dst]
        self.rest_lengths = torch.norm(rest_vec, dim=1) # Shape: (Num_Edges,)

        # 5. Setup Pinned Nodes (Column 0)
        H, W = cfg.HEIGHT, cfg.WIDTH
        pinned_indices = [r * W for r in range(H)]
        print(f"pinned_indices: {pinned_indices}")
        self.pinned_idx = torch.tensor(pinned_indices, dtype=torch.long, device=device)
        self.pinned_pos_target = pos_only[self.pinned_idx] # (N_Pin, 3)

    def de_normalize(self, tensor_norm):
        """Revert standard scaler normalization to get real units (meters/s^2)"""
        return (tensor_norm * self.std) + self.mean

    def compute_chamfer_loss(self, pred_pos, target_pos):
        """
        Computes symmetric Chamfer Distance between two point clouds.
        pred_pos, target_pos: (B, N, 3)
        """
        # 1. Compute pairwise distances (B, N, N)
        # cdist is an optimized C++ implementation in PyTorch
        dists = torch.cdist(pred_pos, target_pos) 
        
        # 2. For each point in pred, find closest point in target
        min_dist_pred, _ = torch.min(dists, dim=2) # (B, N)
        
        # 3. For each point in target, find closest point in pred
        min_dist_target, _ = torch.min(dists, dim=1) # (B, N)
        
        # 4. Average the distances
        return torch.mean(min_dist_pred) + torch.mean(min_dist_target)
    
    def forward(self, pred_norm, target_norm, curr_pos, curr_vel):
        """
        pred_norm: Model Output (Normalized Acceleration) [B, N, 3]
        target_norm: Ground Truth (Normalized Acceleration) [B, N, 3]
        curr_pos: Real-world Position at time t [B, N, 3]
        curr_vel: Real-world Velocity at time t [B, N, 3]
        """
        
        # --- 1. Standard MSE Loss (Supervised) ---
        mse_loss = F.mse_loss(pred_norm, target_norm)

        # --- 2. INTEGRATION (The Critical Step) ---
        # We must de-normalize predictions to apply physics laws
        pred_accel_real = self.de_normalize(pred_norm)

        # Euler Integration: Pos_new = Pos_old + Vel*dt + 0.5*Acc*dt^2
        # This tells us where the nodes WILL be based on the model's prediction
        pred_pos_next = curr_pos + (curr_vel * self.dt) + (0.5 * pred_accel_real * (self.dt ** 2))
        
        target_accel_real = self.de_normalize(target_norm)
        target_pos_next = curr_pos + (curr_vel * self.dt) + (0.5 * target_accel_real * (self.dt ** 2))
        
        
        # Champher Loss
        chamfer_loss = self.compute_chamfer_loss(pred_pos_next, target_pos_next)
        

        # --- EDGE LOSS (Stretch/edge) ---
        # "Don't let the flag turn into spaghetti"
        # We compare the length of edges in pred_pos_next vs rest_lengths
        
        # Get positions of connected nodes
        p_src = pred_pos_next[:, self.src, :] # (B, E, 3)
        p_dst = pred_pos_next[:, self.dst, :] # (B, E, 3)
        
        # Calculate current lengths
        curr_vec = p_src - p_dst
        curr_lengths = torch.norm(curr_vec, dim=2) # (B, E)
        
        # Loss: Difference between current length and rest length
        # We use relative error: |L_curr - L_rest| / L_rest
        # This prevents long edges from dominating the loss
        length_diff = curr_lengths - self.rest_lengths
        edge_loss = torch.mean(length_diff ** 2)

        # --- 4. PIN LOSS (Position Constraint) ---
        # "Don't let the pole move"
        # Instead of just Accel=0, we enforce Pos=Target
        
        current_pinned_pos = pred_pos_next[:, self.pinned_idx, :] # (B, N_Pin, 3)
        # Expand target to batch size
        target_pos_expanded = self.pinned_pos_target.unsqueeze(0).expand(current_pinned_pos.shape)
        
        pin_loss = F.mse_loss(current_pinned_pos, target_pos_expanded)

        # --- 5. Total Loss ---
        total_loss = (self.lambda_rmse * mse_loss) + \
                     (self.lambda_chamfer * chamfer_loss) + \
                     (self.lambda_edge * edge_loss) + \
                     (self.lambda_pin * pin_loss)

        return total_loss, mse_loss, chamfer_loss, edge_loss, pin_loss