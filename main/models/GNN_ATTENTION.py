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

        # ---- NODE & WIND ENCODERS (SEPARATE) ----
        self.node_encoder = MLP(in_node_dim, hidden_dim, hidden_dim)
        self.wind_encoder = MLP(in_wind_dim, hidden_dim, hidden_dim)

        # ---- CROSS ATTENTION (WIND → NODE) ----
        self.wind_to_node_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=cfg.NUM_ATTENTION_HEADS,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ---- EDGE ENCODER ----
        self.edge_encoder = MLP(in_edge_dim, hidden_dim, hidden_dim)

        # ---- GNN PROCESSOR ----
        self.layers = nn.ModuleList([
            InteractionBlock(hidden_dim) for _ in range(num_layers)
        ])

        # ---- DECODER ----
        self.decoder = MLP(hidden_dim, 3, hidden_dim)


    def forward(self, x_nodes, x_wind, edge_index):

        # ---- 1. PHYSICS EDGE FEATURES ----
        row, col = edge_index
        pos = x_nodes[:, :3]
        vel = x_nodes[:, 3:6]

        rel_pos = pos[row] - pos[col]
        rel_vel = vel[row] - vel[col]
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        raw_edge_features = torch.cat([rel_pos, rel_vel, rel_dist], dim=1)
        edge_attr = self.edge_encoder(raw_edge_features)

        # ---- 2. ENCODE NODES & WIND (SEPARATELY) ----
        x = self.node_encoder(x_nodes)     # (N, H)
        w = self.wind_encoder(x_wind)      # (W, H)

        # ---- 3. CROSS ATTENTION (ENCODING STEP) ----
        attn_out, _ = self.wind_to_node_attn(
            query=x.unsqueeze(0),   # nodes = queries
            key=w.unsqueeze(0),     # wind = keys
            value=w.unsqueeze(0)
        )

        x = self.attn_norm(x + attn_out.squeeze(0))

        # ---- 4. GNN MESSAGE PASSING ----
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # ---- 5. DECODE ----
        return self.decoder(x)
