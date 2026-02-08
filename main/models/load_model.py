

import config as cfg

def load_model(device):
    if cfg.MODEL == 'GNN' | cfg.MODEL == 'GNN_ATTENTION':
        from models.GNN import FlagGraphNet as ModelClass
        model = ModelClass(
            in_node_dim=cfg.NODE_DIM,
            in_wind_dim=cfg.WIND_DIM,
            in_edge_dim=cfg.EDGE_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_layers=cfg.NO_GNN_LAYERS
        ).to(device)
        return model
    
    elif cfg.MODEL == 'SNN':
        from models.SNN import FlagWindNet as ModelClass
        model = ModelClass(
            in_node_dim=cfg.NODE_DIM,
            in_wind_dim=cfg.WIND_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_layers=cfg.NO_MLP_HIDDEN_LAYERS
        ).to(device)
        return model
    
    elif cfg.MODEL == 'LSTM_CNN':
        from models.LSTM import FlagLSTM_CNN_Net as ModelClass
        model = ModelClass(
            in_node_dim=cfg.NODE_DIM,
            in_wind_dim=cfg.WIND_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            sequence_length=cfg.SEQUENCE_LENGTH
        ).to(device)
        return model
    
    else:
        raise ValueError(f"Unknown MODEL in config: {cfg.MODEL}")