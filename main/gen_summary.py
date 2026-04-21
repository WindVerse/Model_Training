import config as cfg

def generate_details(train_loss, test_rmse, test_edge_err, time_per_frame, trainable_params):
    # --- General Summary ---
    details = f"""
        **** Results Summary ****
            "Train Loss": {train_loss},
            "Average Test RMSE": {test_rmse},
            "Average Test Edge Error": {test_edge_err},
            "Average Time per Frame": {time_per_frame:.4f} seconds,

        **** Model Configuration ****
            "Model Type": {cfg.MODEL},
            "Batch Size": {cfg.BATCH_SIZE},
            "Epochs": {cfg.EPOCHS},
            "Warmup Epochs": {cfg.WARMUP_EPOCHS},
            "Learning Rate": {cfg.LEARNING_RATE},
            "History Window": {cfg.HISTORY_WINDOW},
            "Sequence Length": {cfg.SEQUENCE_LENGTH},
            "Dropout Rate": {cfg.DROPOUT_RATE},
            "Layer Normalization": {cfg.USE_LAYER_NORM},
            "Total Trainable Parameters": {trainable_params},
    """

    # --- Model Specifics ---
    if cfg.MODEL == 'GNN':
        details += f"""
            "Hidden Dimension": {cfg.HIDDEN_DIM},
            "Number of GNN Layers": {cfg.NO_GNN_LAYERS},
            "GNN Aggregation": {cfg.GNN_AGGREGATION},
            "Number of MLP Hidden Layers": {cfg.NO_MLP_HIDDEN_LAYERS},
            "Activation Function": {cfg.ACTIVATION},
        """
    elif cfg.MODEL == 'SNN':
        details += f"""
            "Hidden Dimension": {cfg.HIDDEN_DIM},
            "Number of MLP Hidden Layers": {cfg.NO_MLP_HIDDEN_LAYERS},
            "Activation Function": {cfg.ACTIVATION},
        """
    elif cfg.MODEL == 'LSTM_CNN':
        details += f"""
            "Hidden Dimension": {cfg.HIDDEN_DIM},
            "Number of LSTM Layers": {cfg.NO_LSTM_LAYERS},
            "Number of CNN Layers": {len(cfg.CNN_CHANNELS)},
            "CNN Channels": {cfg.CNN_CHANNELS},
            "Activation Function": {cfg.ACTIVATION},
        """

    # --- Loss Configuration ---
    # Always add the header first
    details += "\n        **** Loss Configuration ****"
    
    if cfg.LOSS == 'physicsLoss':
        details += f"""
            "Loss Function": {cfg.LOSS},
            "Lambda RMSE": {cfg.LAMBDA_RMSE},
            "Lambda Positional": {cfg.LAMBDA_POSITIONAL},
            "Lambda Chamfer": {cfg.LAMBDA_CHAMFER},
            "Lambda EDGE": {cfg.LAMBDA_EDGE},
            "Lambda Smooth": {cfg.LAMBDA_SMOOTH},
            "Lambda Area": {cfg.LAMBDA_AREA},
            "Lambda Bend": {cfg.LAMBDA_BEND},
            "Lambda Pin": {cfg.LAMBDA_PIN},
        """
    else:
        # Handle standard losses (MSE, L1, etc)
        details += f"""
            "Loss Function": {cfg.LOSS},
        """

    # --- Optimizer Configuration ---
    details += "\n        **** Optimizer Configuration ****"

    if cfg.OPTIMIZER == 'Adam':
        details += f"""
            "Optimizer": {cfg.OPTIMIZER},
            "Learning Rate": {cfg.LEARNING_RATE},
            "Weight Decay": {cfg.WEIGHT_DECAY},
        """
    elif cfg.OPTIMIZER == 'SGD':
        details += f"""
            "Optimizer": {cfg.OPTIMIZER},
            "Learning Rate": {cfg.LEARNING_RATE},
            "Momentum": {cfg.MOMENTUM},
            "Weight Decay": {cfg.WEIGHT_DECAY},
        """
    elif cfg.OPTIMIZER == 'RMSprop':
        details += f"""
            "Optimizer": {cfg.OPTIMIZER},
            "Learning Rate": {cfg.LEARNING_RATE},
            "Weight Decay": {cfg.WEIGHT_DECAY},
            "Momentum": {getattr(cfg, 'MOMENTUM', 0)}, 
        """
    else:
        details += f"""
            "Optimizer": {cfg.OPTIMIZER},
        """

    # --- Scheduler Configuration ---
    details += "\n        **** Scheduler Configuration ****"

    if cfg.SCHEDULER == 'ReduceLROnPlateau':
        details += f"""
            "Scheduler": {cfg.SCHEDULER},
            "Mode": {cfg.SCHEDULER_MODE},
            "Factor": {cfg.SCHEDULER_FACTOR},
            "Patience": {cfg.SCHEDULER_PATIENCE},
        """
    elif cfg.SCHEDULER == 'StepLR':
        details += f"""
            "Scheduler": {cfg.SCHEDULER},
            "Step Size": {cfg.SCHEDULER_STEP_SIZE},
            "Gamma": {cfg.SCHEDULER_GAMMA},
        """
    else:
        details += f"""
            "Scheduler": {cfg.SCHEDULER},
        """
    
    # --- Noise Configuration ---
    details += "\n        **** Noise Configuration ****"
    details += f"""
        "Add Noise": {cfg.ADD_NOISE},
        "Noise Std Dev": {cfg.NOISE_STD},
    """
    
    # --- Other Configurations ---
    details += "\n        **** Other Configurations ****"
    details += f"""
        "Target": {cfg.TARGET_TYPE},
        "Validating": {cfg.VALIDATE},
    """

    return details