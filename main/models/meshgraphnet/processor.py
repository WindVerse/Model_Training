import torch
from torch_geometric.nn import MessagePassing
from models.meshgraphnet.mlp import MeshGraphMLP

class ProcessorBlock(MessagePassing):
    def __init__(self, latent_dim=128):
        # We use 'sum' aggregation for physics (aggregating all incoming forces)
        super().__init__(aggr='sum') 
        
        # MLP to update the edge latent space
        self.edge_mlp = MeshGraphMLP(latent_dim * 3, latent_dim) 
        
        # MLP to update the node latent space
        self.node_mlp = MeshGraphMLP(latent_dim * 2, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        # 1. EDGE UPDATE
        # Gather sender (row) and receiver (col) nodes
        # Use explicit slicing for TorchScript compatibility
        row = edge_index[0]
        col = edge_index[1]
        # Concatenate: [Sender Node, Receiver Node, Current Edge]
        edge_inputs = torch.cat([x[row], x[col], edge_attr], dim=-1)
        # Pass through MLP to get new edge features
        updated_edges = self.edge_mlp(edge_inputs)
        
        # 2. NODE UPDATE (Message Passing)
        # Propagate automatically calls message(), aggregate(), and update()
        updated_nodes = self.propagate(edge_index, x=x, updated_edges=updated_edges)
        
        # 3. RESIDUAL CONNECTIONS (Critical for avoiding vanishing gradients across 15 layers)
        return x + updated_nodes, edge_attr + updated_edges

    def message(self, updated_edges):
        # The message passed to the aggregating node IS the updated edge feature
        return updated_edges

    def update(self, aggr_out, x):
        # aggr_out is the SUM of all incoming updated_edges.
        # Concatenate the node's current state with the aggregated messages
        node_inputs = torch.cat([x, aggr_out], dim=-1)
        # Pass through MLP to get new node features
        return self.node_mlp(node_inputs)