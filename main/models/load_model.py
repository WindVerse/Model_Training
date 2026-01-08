

import config as cfg

def load_model(device):
    if cfg.MODEL == 'GNN':
        from models.GNN import FlagGraphNet as ModelClass
        model = ModelClass(
            in_node_dim=cfg.NODE_DIM,
            in_wind_dim=cfg.WIND_DIM,
            in_edge_dim=cfg.EDGE_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_layers=cfg.NO_GNN_LAYERS
        ).to(device)
        return model
    else:
        raise ValueError(f"Unknown MODEL in config: {cfg.MODEL}")