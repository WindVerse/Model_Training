import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg

class PhysicsLoss(nn.Module):
    def __init__(self, 
                 initial_flag_pos,  # Can be (N, 6) or (N, 3) numpy array
                 mean,              # Training set Mean (tensor or array)
                 std,               # Training set Std (tensor or array)
                 device="cuda"):
        super().__init__()

        # 1. Hyperparameters from Config
        self.lambda_warp = cfg.LAMBDA_WARP
        self.lambda_smooth = cfg.LAMBDA_SMOOTH
        self.lambda_pin = cfg.LAMBDA_PIN

        # 2. Store Normalization Stats (Needed to de-normalize for physics calc)
        # Reshape to (1, 1, Features) for broadcasting
        self.mean = torch.as_tensor(mean, device=device).view(1, 1, -1)
        self.std = torch.as_tensor(std, device=device).view(1, 1, -1)

        # 3. Load Topology (Edges)
        edge_index = np.load(cfg.TOPOLOGY_PATH)
        # Store as LongTensor for indexing
        self.src = torch.from_numpy(edge_index[0]).long().to(device)
        self.dst = torch.from_numpy(edge_index[1]).long().to(device)

        # 4. Setup Rest Lengths (The "Springs")
        initial_pos = torch.as_tensor(initial_flag_pos, device=device).float()
        
        # Take only Position (first 3 channels), ignore Velocity
        pos_only = initial_pos[:, :3] 

        # Calculate vector between connected nodes
        rest_vec = pos_only[self.src] - pos_only[self.dst]
        
        # Calculate Euclidean distance (L2 norm)
        # Shape: (Num_Edges,)
        self.rest_lengths = torch.norm(rest_vec, dim=1)

        # 5. Setup Pinned Nodes (The "Pole")
        # Automatically detect pinned nodes: Column 0 of the grid
        # In a row-major grid of W columns, indices are 0, W, 2W, 3W...
        H, W = cfg.HEIGHT, cfg.WIDTH
        
        # Generate indices for the first column (0, 30, 60...)
        pinned_indices = [r * W for r in range(H)]
        
        self.pinned_idx = torch.tensor(pinned_indices, dtype=torch.long, device=device)
        
        # Store the EXACT Target Position for these pinned nodes
        # Shape: (Num_Pinned, 3)
        self.pinned_pos_target = pos_only[self.pinned_idx]

    def forward(self, pred_norm, target_norm):
        """
        pred_norm: Model Output (Normalized Acceleration)
        target_norm: Ground Truth (Normalized Acceleration)
        NOTE: Ideally, physics loss works on POSITION. 
        If your model outputs acceleration, you assume standard MSE is dominant, 
        and these auxiliary losses help regularize the latent physics.
        """
        
        # 1. Flatten Batch and Time dimensions for simpler processing
        # Input: (Batch, Time, Nodes, 3) -> (Batch*Time, Nodes, 3)
        if pred_norm.dim() == 4:
            B, L, N, D = pred_norm.shape
            pred_flat = pred_norm.reshape(B * L, N, D)
            target_flat = target_norm.reshape(B * L, N, D)
        else:
            pred_flat = pred_norm
            target_flat = target_norm

        # --- A. Standard MSE Loss (Normalized Space) ---
        mse_loss = F.mse_loss(pred_flat, target_flat)

        # --- PREPARE REAL WORLD DATA (De-normalization) ---
        # NOTE: If your model predicts ACCELERATION, calculating position constraints
        # is tricky without a full integration step. 
        # HOWEVER, we can apply these constraints to the "Predicted Next State" 
        # if we had the previous state.
        #
        # Since this loss function only receives `pred` (acceleration), 
        # we will apply the constraints to the acceleration vector itself 
        # (e.g. pinned nodes should have 0 acceleration).
        
        # --- B. PIN LOSS (Acceleration Constraint) ---
        # Pinned nodes should have ZERO acceleration (stay still).
        # We don't need de-normalization for this; 0 is 0.
        
        pred_accel = pred_flat # (Batch, Nodes, 3)
        
        # Extract acceleration of pinned nodes
        current_pinned_accel = pred_accel[:, self.pinned_idx] # (Batch, N_Pin, 3)
        
        # Target is Zero Acceleration
        target_pinned_accel = torch.zeros_like(current_pinned_accel)
        
        pin_loss = F.mse_loss(current_pinned_accel, target_pinned_accel)

        # --- TOTAL LOSS ---
        # Note: We skipped Edge Loss here because calculating edge length from 
        # acceleration alone is mathematically invalid without the velocity/position context.
        # Ideally, you integrate (Pos_new = Pos_old + Vel*dt + 0.5*Acc*dt^2) 
        # and apply Edge Loss on Pos_new.
        
        total_loss = mse_loss + (self.lambda_pin * pin_loss)

        return total_loss, mse_loss, pin_loss