import torch
import torch.nn as nn
from models.meshgraphnet.mlp import MeshGraphMLP

class Encoder(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, latent_dim=128):
        super().__init__()
        
        # DYNAMIC NORMALIZERS: Replaces dataset.py normalization
        self.node_normalizer = nn.LayerNorm(node_input_dim)
        self.edge_normalizer = nn.LayerNorm(edge_input_dim)
        
        # Latent Encoders
        self.node_encoder = MeshGraphMLP(node_input_dim, latent_dim)
        self.edge_encoder = MeshGraphMLP(edge_input_dim, latent_dim)

    def forward(self, x, edge_attr):
        # 1. Normalize raw physical inputs on the fly
        x_norm = self.node_normalizer(x)
        edge_attr_norm = self.edge_normalizer(edge_attr)
        
        # 2. Project to 128-dimensional latent space
        node_latent = self.node_encoder(x_norm)
        edge_latent = self.edge_encoder(edge_attr_norm)
        
        return node_latent, edge_latent