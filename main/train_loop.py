import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

import config as cfg
from validateVis import validate_rollout

def get_next_version_dir(base_dir):
    """
    Scans base_dir for folders like '001', '002' and returns the next available path.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    numeric_dirs = []
    for d in existing_dirs:
        try:
            numeric_dirs.append(int(d))
        except ValueError:
            continue
            
    next_val = max(numeric_dirs) + 1 if numeric_dirs else 1
    version_dir = os.path.join(base_dir, f"{next_val:03d}")
    os.makedirs(version_dir, exist_ok=True)
    print(f"📂 Output directory created: {version_dir}")
    return version_dir

def export_onnx(model, save_path, device):
    """
    Exports the model to ONNX format with dynamic axes.
    """
    model.eval()
    
    # Dummy Input
    dummy_nodes = 100 
    dummy_edges = 200
    
    x_nodes = torch.randn(dummy_nodes, cfg.NODE_DIM, device=device)
    x_wind = torch.randn(dummy_nodes, cfg.WIND_DIM, device=device)
    edge_index = torch.randint(0, dummy_nodes, (2, dummy_edges), device=device).long()
    
    input_names = ["x_nodes", "x_wind", "edge_index"]
    output_names = ["acceleration"]
    
    dynamic_axes = {
        "x_nodes": {0: "num_nodes"},
        "x_wind": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "acceleration": {0: "num_nodes"}
    }
    
    try:
        torch.onnx.export(
            model,
            (x_nodes, x_wind, edge_index),
            save_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print(f"📦 ONNX Model exported to: {save_path}")
    except Exception as e:
        print(f"⚠️ ONNX Export Failed: {e}")

def setup_optimization(model):
    """Helper to initialize Optimizer and Scheduler."""
    if cfg.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown OPTIMIZER: {cfg.OPTIMIZER}")

    scheduler = None
    if cfg.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.SCHEDULER_FACTOR, patience=cfg.SCHEDULER_PATIENCE)
    elif cfg.SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SCHEDULER_STEP_SIZE, gamma=cfg.SCHEDULER_GAMMA)
    elif cfg.SCHEDULER == 'None':
        scheduler = None
    else:
        raise ValueError(f"Unknown SCHEDULER: {cfg.SCHEDULER}")

    return optimizer, scheduler

def trainModel(train_set, test_set, device):
    """Main training loop (Validation Removed)."""
    
    # ==========================================
    # 0. SETUP OUTPUT DIRECTORY
    # ==========================================
    models_root = os.path.join(cfg.DATASET_DIR, "models")
    run_dir = get_next_version_dir(models_root)

    # DATA LOADER (Only Train)
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    # ==========================================
    # 1. MODEL INITIALIZATION
    # ==========================================
    if cfg.MODEL == 'GNN':
        from models.GNN import FlagGraphNet as ModelClass
        model = ModelClass(
            in_node_dim=cfg.NODE_DIM,
            in_wind_dim=cfg.WIND_DIM,
            in_edge_dim=cfg.EDGE_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_layers=cfg.NO_GNN_LAYERS
        ).to(device)
    else:
        raise ValueError(f"Unknown MODEL in config: {cfg.MODEL}")
    
    print(f"Initializing Model: {cfg.MODEL}...")

    # ==========================================
    # 2. LOSS INITIALIZATION
    # ==========================================
    print(f"Initializing Loss: {cfg.LOSS}...")
    
    if cfg.LOSS == 'physicsLoss':
        from loss.physics_loss import PhysicsLoss
        initial_pos_ref = train_set.data_flags[0][0, :, :3] 
        criterion = PhysicsLoss(
            initial_flag_pos=initial_pos_ref,
            mean=train_set.stats['target_mean'],
            std=train_set.stats['target_std'],
            device=device
        )
    else:
        raise ValueError(f"Unknown LOSS in config: {cfg.LOSS}")

    # ==========================================
    # 3. OPTIMIZER & SCHEDULER SETUP
    # ==========================================
    optimizer, scheduler = setup_optimization(model)
    print(f"Optimizer: {cfg.OPTIMIZER} | Scheduler: {cfg.SCHEDULER}")

    # Load Topology (Edges)
    if os.path.exists(cfg.TOPOLOGY_PATH):
        edge_index_np = np.load(cfg.TOPOLOGY_PATH)
        base_edge_index = torch.from_numpy(edge_index_np).long().to(device)
    else:
        raise FileNotFoundError(f"Topology not found at {cfg.TOPOLOGY_PATH}")

    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    best_loss = float('inf')  # Track Training Loss
    print(f"Starting training for {cfg.EPOCHS} epochs...")
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_train_loss = 0
        total_mse = 0
        total_pin = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        
        for batch_idx, (flag_seq, wind_seq, target_seq) in enumerate(loop):
            # Move to device
            flag_seq = flag_seq.to(device)
            wind_seq = wind_seq.to(device)
            target_seq = target_seq.to(device)

            # --- PREPARE DATA FOR GNN ---
            B, T, N, F = flag_seq.shape
            
            x_nodes = flag_seq.view(B * T, N, F)
            x_wind_raw = wind_seq.view(B * T, 8, 3)
            y_target = target_seq.view(B * T, N, 3)

            x_wind_mean = x_wind_raw.mean(dim=1) 
            x_wind_expanded = x_wind_mean.unsqueeze(1).repeat(1, N, 1)

            x_nodes_flat = x_nodes.view(-1, F)
            x_wind_flat = x_wind_expanded.view(-1, 3)
            y_target_flat = y_target.view(-1, 3)

            # --- PREPARE BATCH EDGES ---
            edge_index_batch = []
            for i in range(B * T):
                edge_index_batch.append(base_edge_index + (i * N))
            edge_index_flat = torch.cat(edge_index_batch, dim=1)

            # --- FORWARD STEP ---
            optimizer.zero_grad()
            pred_accel = model(x_nodes_flat, x_wind_flat, edge_index_flat)

            # Reshape for Loss
            pred_reshaped = pred_accel.view(B*T, N, 3)
            target_reshaped = y_target_flat.view(B*T, N, 3)
            
            loss, mse, pin_loss = criterion(pred_reshaped, target_reshaped)

            # --- BACKWARD STEP ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            total_train_loss += loss.item()
            total_mse += mse.item()
            total_pin += pin_loss.item()
            loop.set_postfix(loss=loss.item(), mse=mse.item(), pin=pin_loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ==========================================
        # 5. SCHEDULER & SAVING (Using Train Loss)
        # ==========================================
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.5f} | LR: {current_lr:.6f}")

        if scheduler is not None:
            if cfg.SCHEDULER == 'ReduceLROnPlateau':
                # Step based on Training Loss now
                scheduler.step(avg_train_loss)
            else:
                scheduler.step()

        # Save Best Model (Based on Train Loss)
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            
            # Paths
            pth_path = os.path.join(run_dir, "best_model.pth")
            onnx_path = os.path.join(run_dir, "best_model.onnx")
            
            # 1. Save PTH
            torch.save(model.state_dict(), pth_path)
            print(f"💾 Best Model Saved: {pth_path}")
            
            # 2. Save ONNX
            export_onnx(model, onnx_path, device)

    # ==========================================
    # 6. RELOAD BEST WEIGHTS
    # ==========================================
    print("Training Complete. Reloading best model...")
    best_pth_path = os.path.join(run_dir, "best_model.pth")
    
    # Fixed weights_only parameter (it is 'weights_only', not 'weight_only')
    model.load_state_dict(torch.load(best_pth_path, weights_only=True))
    
    # ==========================================
    # TRIGGER VISUAL VALIDATION
    # ==========================================
    model_ver = os.path.basename(run_dir)
    print(f"🎨 Triggering Visualization for Model Version: {model_ver}")

    # Extract UNIQUE Run IDs from the test set
    unique_test_runs = sorted(list(set([sample[0] for sample in test_set.samples])))
    
    print(f"Found {len(unique_test_runs)} unique runs in Test Set: {unique_test_runs}")

    # 2. Iterate over Unique Runs only
    for run_idx in unique_test_runs:
        validate_rollout(
            dataset=test_set, 
            model_ver=model_ver, 
            run_index=run_idx
        )
    
    print(f"✅ Returned model with Best Train Loss: {best_loss:.5f}")
    return model