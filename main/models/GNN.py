import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

import config as cfg

def get_activation(name):
    """Helper to map config string to PyTorch class"""
    if name == 'ReLU': return nn.ReLU()
    if name == 'SiLU': return nn.SiLU()
    if name == 'Tanh': return nn.Tanh()
    if name == 'LeakyReLU': return nn.LeakyReLU()
    raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=cfg.HIDDEN_DIM, num_layers=cfg.NO_MLP_HIDDEN_LAYERS):
        super().__init__()
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        if cfg.USE_LAYER_NORM: layers.append(nn.LayerNorm(hidden_dim))
        layers.append(get_activation(cfg.ACTIVATION))
        layers.append(nn.Dropout(cfg.DROPOUT_RATE)) # NEW: Dropout
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if cfg.USE_LAYER_NORM: layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(cfg.ACTIVATION))
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

    def forward(self, x_nodes, x_wind, edge_index):
        # 1. Feature Engineering
        row, col = edge_index
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
        return self.decoder(x)