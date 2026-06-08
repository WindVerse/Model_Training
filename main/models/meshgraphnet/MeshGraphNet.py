import torch
import torch.nn as nn
from models.meshgraphnet.encoder import Encoder
from models.meshgraphnet.processor import ProcessorBlock
from models.meshgraphnet.decoder import Decoder

class MeshGraphNet(nn.Module):
    def __init__(self, node_input_dim=7, edge_input_dim=8, latent_dim=128, num_processor_steps=15, output_dim=3):
        super().__init__()
        
        # 1. Encoder
        self.encoder = Encoder(node_input_dim, edge_input_dim, latent_dim)
        
        # 2. 15 Processor Blocks
        self.processor_blocks = nn.ModuleList([
            ProcessorBlock(latent_dim) for _ in range(num_processor_steps)
        ])
        
        # 3. Decoder
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x_nodes, edge_index, edge_attr):
        # Step 1: Encode into 128-dimensional space
        x, edge_attr = self.encoder(x_nodes, edge_attr)
        
        # Step 2: Message Passing sequence
        for block in self.processor_blocks:
            x, edge_attr = block(x, edge_index, edge_attr)
            
        # Step 3: Decode back to 3D Physical Space
        out = self.decoder(x)
        
        return out