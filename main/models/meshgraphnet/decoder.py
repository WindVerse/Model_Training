import torch.nn as nn
from models.meshgraphnet.mlp import MeshGraphMLP

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=3):
        super().__init__()
        # layer_norm=False because we want raw continuous outputs, not bounded ones.
        self.node_decoder = MeshGraphMLP(latent_dim, output_dim, layer_norm=False)

    def forward(self, x):
        return self.node_decoder(x)