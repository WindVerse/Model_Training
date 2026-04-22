import torch.nn as nn

import config as cfg

class MeshGraphMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, hidden_layers=cfg.NO_MLP_HIDDEN_LAYERS, layer_norm=True):
        super().__init__()
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden Layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Layer Normalization (DeepMind applies this to all MLPs except the final Decoder)
        if layer_norm:
            layers.append(nn.LayerNorm(output_dim))
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)