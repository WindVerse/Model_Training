import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

import config as cfg
import models.model_helpers.model_helpers as helpers

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=cfg.HIDDEN_DIM, num_layers=cfg.NO_MLP_HIDDEN_LAYERS):
        super().__init__()
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        if cfg.USE_LAYER_NORM: layers.append(nn.LayerNorm(hidden_dim))
        layers.append(helpers.get_activation(cfg.ACTIVATION))
        layers.append(nn.Dropout(cfg.DROPOUT_RATE)) # NEW: Dropout
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if cfg.USE_LAYER_NORM: layers.append(nn.LayerNorm(hidden_dim))
            layers.append(helpers.get_activation(cfg.ACTIVATION))
            layers.append(nn.Dropout(cfg.DROPOUT_RATE)) # NEW: Dropout
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

class InteractionBlock(MessagePassing):
    def __init__(self, hidden_dim=cfg.HIDDEN_DIM):
        # Use config for aggregation ('add' is standard for forces)
        super().__init__(aggr=cfg.GNN_AGGREGATION) 
        
        self.edge_mlp = MLP(in_dim=hidden_dim*3, out_dim=hidden_dim)
        self.node_mlp = MLP(in_dim=hidden_dim*2, out_dim=hidden_dim)
        
        self.res_norm = nn.LayerNorm(hidden_dim) if cfg.USE_LAYER_NORM else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        latent_update = self.node_mlp(tmp)
        return self.res_norm(x + latent_update)

class FlagGraphNet(nn.Module):
    def __init__(self, 
                 in_node_dim=cfg.NODE_DIM, 
                 in_wind_dim=cfg.WIND_DIM, 
                 in_edge_dim=cfg.EDGE_DIM,
                 hidden_dim=cfg.HIDDEN_DIM, 
                 num_layers=cfg.NO_GNN_LAYERS):
        super().__init__()
        
        # Encoders
        self.node_encoder = MLP(in_node_dim + in_wind_dim, hidden_dim, hidden_dim)
        self.edge_encoder = MLP(in_edge_dim, hidden_dim, hidden_dim)

        # Processor
        self.layers = nn.ModuleList([
            InteractionBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = MLP(hidden_dim, 3, hidden_dim) 
        
        # ====================================================
        # INTERNAL MASK GENERATION (HARD PINNING)
        # ====================================================
        H, W = cfg.HEIGHT, cfg.WIDTH
        self.num_nodes_per_flag = H * W
        
        # 1. Create Base Mask (N, 1)
        # 1.0 = Pinned (Acc forced to 0), 0.0 = Free
        mask = torch.zeros((self.num_nodes_per_flag, 1))
        
        # Pin Column 0 (Indices: 0, W, 2W...)
        # This matches the "Row-Major" flattening logic
        for r in range(H):
            idx = r * W
            mask[idx, 0] = 1.0
            
        # 2. Register as buffer
        # This saves 'pinned_mask' to state_dict and moves it to device automatically
        self.register_buffer('pinned_mask', mask)

    def forward(self, x_nodes, x_wind, edge_index):
        # 1. Feature Engineering
        row = edge_index[0]
        col = edge_index[1]
        pos = x_nodes[:, :3]
        vel = x_nodes[:, 3:6]
        
        rel_pos = pos[row] - pos[col]
        rel_vel = vel[row] - vel[col]
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # Concatenate explicit physics features
        raw_edge_features = torch.cat([rel_pos, rel_vel, rel_dist], dim=1)
        
        # 2. Encode
        combined_node = torch.cat([x_nodes, x_wind], dim=1)
        x = self.node_encoder(combined_node)
        edge_attr = self.edge_encoder(raw_edge_features)

        # 3. Process
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # 4. Decode
        out = self.decoder(x)
        
        
        # 5. Hard Pinning
        
        current_batch_nodes = x_nodes.shape[0]
        num_flags_in_batch = current_batch_nodes // self.num_nodes_per_flag
        
        # Repeat mask: (N, 1) -> (Batch*N, 1)
        full_mask = self.pinned_mask.repeat(num_flags_in_batch, 1)
        
        # Apply: (1.0 - 1.0) = 0.0 for Pinned
        #        (1.0 - 0.0) = 1.0 for Free
        out = out * (1.0 - full_mask)
        
        return out