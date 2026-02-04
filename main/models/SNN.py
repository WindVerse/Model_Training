import torch
import torch.nn as nn
import config as cfg
import models.model_helpers.model_helpers as helpers

class FlagWindNet(nn.Module):
    def __init__(self, 
                 in_node_dim=cfg.NODE_DIM, 
                 in_wind_dim=cfg.WIND_DIM, 
                 hidden_dim=cfg.HIDDEN_DIM, 
                 num_layers=cfg.NO_MLP_HIDDEN_LAYERS):
        super().__init__()
        
        # Flag MLP
        self.flag_mlp = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            helpers.get_activation(cfg.ACTIVATION),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                helpers.get_activation(cfg.ACTIVATION)
              ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Wind MLP
        self.wind_mlp = nn.Sequential(
            nn.Linear(in_wind_dim, hidden_dim),
            helpers.get_activation(cfg.ACTIVATION),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                helpers.get_activation(cfg.ACTIVATION)
              ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            helpers.get_activation(cfg.ACTIVATION),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                helpers.get_activation(cfg.ACTIVATION)
              ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, flag_x, wind_x, edge_index=None):
        """
        Modified to accept 2D Flattened Inputs from train_loop.py
        flag_x: (Total_Batch_Nodes, Node_Dim) e.g., (24000, 6)
        wind_x: (Total_Batch_Nodes, Wind_Dim) e.g., (24000, 3)
        """
        
        # 1. Encode separately
        flag_latent = self.flag_mlp(flag_x) # Output: (Total_Nodes, Hidden)
        wind_latent = self.wind_mlp(wind_x) # Output: (Total_Nodes, Hidden)
        
        # 2. Fuse
        # Since 'wind_x' coming from train_loop is already expanded per-node,
        # we can simply concatenate them. No need for unsqueeze/repeat.
        fused = torch.cat([flag_latent, wind_latent], dim=-1)
        
        # 3. Decode
        out = self.output_mlp(fused)
        
        return out