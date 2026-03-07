import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False

import config as cfg
from validate.validateVis import validate_rollout
from validate.validateMetric import validate_metrics
from models.load_model import load_model
from gen_summary import generate_details
from loss.get_loss import getLoss


def count_parameters(model):
    """Returns the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_architecture_diagram(model, save_path, device):
    """
    Traces the model with dummy inputs and saves a PNG and PDF diagram.
    """
    if not TORCHVIZ_AVAILABLE:
        print("⚠️ torchviz not found. Skipping architecture diagram.")
        return

    model.eval()
    
    try:
        # 1. Create Dummy Inputs based on Model Type
        if cfg.MODEL == 'GNN':
            # GNN inputs: Nodes, Wind, Edges
            N_dummy = cfg.HEIGHT * cfg.WIDTH 
            E_dummy = 2 * ((cfg.HEIGHT - 1) * cfg.WIDTH + (cfg.WIDTH - 1) * cfg.HEIGHT)

            x_nodes = torch.randn(N_dummy, cfg.NODE_DIM).to(device)
            x_wind = torch.randn(N_dummy, cfg.WIND_DIM).to(device)
            edge_index = torch.randint(0, N_dummy, (2, E_dummy)).to(device)
            
            # Trace
            y = model(x_nodes, x_wind, edge_index)

        elif cfg.MODEL == 'SNN':
            # SNN inputs: Flattened (Node + Wind)
            B_dummy = 1
            input_dim = cfg.NODE_DIM + cfg.WIND_DIM
            x = torch.randn(B_dummy, input_dim).to(device)
            
            # Trace
            y = model(x)

        elif 'LSTM' in cfg.MODEL:
            # LSTM inputs: Sequence (Batch, Seq, Features)
            B_dummy = 1
            S_dummy = cfg.SEQUENCE_LENGTH
            N_real = cfg.HEIGHT * cfg.WIDTH
            
            # Create Dummy Nodes and Wind separately
            dummy_nodes = torch.randn(B_dummy * S_dummy * N_real, cfg.NODE_DIM).to(device)
            dummy_wind = torch.randn(B_dummy * S_dummy * N_real, cfg.WIND_DIM).to(device)
            
            # Trace (Unpack tuple!)
            y, _ = model(dummy_nodes, dummy_wind)
        
        else:
            return # Unknown model type, skip viz

        # 2. Generate and Save Plot
        dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        
        # --- SAVE AS PNG ---
        dot.format = 'png'
        # cleanup=False keeps the source .gv file (the "3rd" file)
        dot.render(save_path, cleanup=False) 
        print(f"📸 Architecture saved to: {save_path}.png")

        # --- SAVE AS PDF (Vector Graphic) ---
        dot.format = 'pdf'
        dot.render(save_path)
        print(f"📸 Architecture saved to: {save_path}.pdf")

    except Exception as e:
        print(f"⚠️ Architecture visualization failed: {e}")
        # Hint for Windows users specifically
        if "dot" in str(e):
            print("💡 Tip: Ensure Graphviz executable is in your system PATH.")
            
    finally:
        model.train() # Switch back to train mode

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
    dummy_nodes = cfg.HEIGHT * cfg.WIDTH 
    dummy_edges = 2 * ((cfg.HEIGHT - 1) * cfg.WIDTH + (cfg.WIDTH - 1) * cfg.HEIGHT)
    
    x_nodes = torch.randn(dummy_nodes, cfg.NODE_DIM, device=device)
    x_wind = torch.randn(dummy_nodes, cfg.WIND_DIM, device=device)
    edge_index = torch.randint(0, dummy_nodes, (2, dummy_edges), device=device).long()
    
    input_names = ["flag", "wind", "edges"]
    output_names = ["output"]
    
    dynamic_axes = {
        "flag": {0: "num_nodes"},
        "wind": {0: "num_nodes"},
        "edges": {1: "num_edges"},
        "output": {0: "num_nodes"}
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg.SCHEDULER_MODE, factor=cfg.SCHEDULER_FACTOR, patience=cfg.SCHEDULER_PATIENCE)
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
    print(f"Initializing Model: {cfg.MODEL}...")
    model = load_model(device)
    total_params = count_parameters(model)
    print(f"Model initialized with {total_params} trainable parameters.")
    save_architecture_diagram(model, os.path.join(run_dir, f"{cfg.MODEL}_architecture"), device)  

    # ==========================================
    # 2. LOSS INITIALIZATION
    # ==========================================
    print(f"Initializing Loss: {cfg.LOSS}...")
    criterion = getLoss(train_set, device)

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
    
    total_lost_history = []
    rmse_history = []
    edge_history = []
    area_history = []
    bend_history = []
    pin_history = []
    chamfer_loss_history = []

    validation_rmse_history = []
    validation_edge_history = []
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_train_loss = 0
        total_rmse = 0
        total_pin = 0
        total_edge = 0
        total_area = 0
        total_bend = 0
        total_chamfer = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        
        for batch_idx, (flag_seq, wind_seq, target_seq) in enumerate(loop):
            # Move to device
            flag_seq = flag_seq.to(device)
            wind_seq = wind_seq.to(device)
            target_seq = target_seq.to(device)

            # --- PREPARE DATA FOR MODEL ---
            B, T, N, F = flag_seq.shape
                        
            x_nodes = flag_seq.view(B * T, N, F)
            x_wind_raw = wind_seq.view(B * T, 8, 3)
            y_target = target_seq.view(B * T, N, 3)

            curr_pos = flag_seq[..., :3].view(B*T, N, 3)

            # curr_pos shape: (B*T, N, 3)
            x = curr_pos[..., 0]
            y = curr_pos[..., 1]
            z = curr_pos[..., 2]

            # Determine which half each coordinate is in
            ix = (x >= 0).long()   # 0 if negative, 1 if positive
            iy = (y >= 0).long()
            iz = (z >= 0).long()

            # Convert 3D index → 1D index (0 to 7)
            cube_index = ix*4 + iy*2 + iz

            # Expand cube_index for gathering
            cube_index_expanded = cube_index.unsqueeze(-1).expand(-1, -1, 3)

            x_wind_expanded = torch.gather(
                x_wind_raw,
                1,
                cube_index_expanded
            )
                   
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
            
            # 1. Call Model
            out = model(x_nodes_flat, x_wind_flat, edge_index_flat)
            
            # 2. Handle Tuple Return (for LSTM) vs Tensor Return (for GNN/SNN)
            if isinstance(out, tuple):
                pred_accel, _ = out # Discard hidden state during training
            else:
                pred_accel = out

            # Reshape for Loss
            pred_reshaped = pred_accel.view(B*T, N, 3)
            target_reshaped = y_target_flat.view(B*T, N, 3)
            
            curr_pos = flag_seq[..., :3].view(B*T, N, 3)
            curr_vel = flag_seq[..., 3:6].view(B*T, N, 3)
            
            loss, rmse, chamfer_loss, edge_loss, area_loss, bend_loss, pin_loss = criterion(pred_reshaped, target_reshaped, curr_pos, curr_vel)

            # --- BACKWARD STEP ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            total_train_loss += loss.item()
            total_rmse += rmse.item()
            total_edge += edge_loss.item()
            total_area += area_loss.item()
            total_bend += bend_loss.item()
            total_pin += pin_loss.item()
            total_chamfer += chamfer_loss.item()
            loop.set_postfix(loss=loss.item(), rmse=rmse.item(), edge=edge_loss.item(), area=area_loss.item(), bend=bend_loss.item(), pin=pin_loss.item(), chamfer=chamfer_loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_rmse = total_rmse / len(train_loader)
        avg_pos = total_chamfer / len(train_loader)
        avg_edge = total_edge / len(train_loader)
        avg_area = total_area / len(train_loader)
        avg_bend = total_bend / len(train_loader)
        avg_pin = total_pin / len(train_loader)
        
        total_lost_history.append(avg_train_loss)
        rmse_history.append(avg_rmse * cfg.LAMBDA_RMSE)
        chamfer_loss_history.append(avg_pos * cfg.LAMBDA_CHAMFER)
        edge_history.append(avg_edge * cfg.LAMBDA_EDGE)
        area_history.append(avg_area * cfg.LAMBDA_AREA)
        bend_history.append(avg_bend * cfg.LAMBDA_BEND)
        pin_history.append(avg_pin * cfg.LAMBDA_PIN)
        

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
        
        # Validate after each epoch
        unique_test_runs = sorted(list(set([sample[0] for sample in test_set.samples])))
        current_epoch_rmse = []
        current_epoch_edge = []
        
        for run_idx in unique_test_runs:
            epoch_rmse, epoch_edge_err, _ = validate_metrics(
                dataset=test_set,
                model_ver=os.path.basename(run_dir),
                run_index=run_idx,
                sub_dir=f"epoch_{epoch+1}",
                model=model
            )
            current_epoch_rmse.append(epoch_rmse)
            current_epoch_edge.append(epoch_edge_err)
        
        # 3. Average and Append to History (ONE value per epoch)
        avg_val_rmse = np.mean(current_epoch_rmse)
        avg_val_edge = np.mean(current_epoch_edge)
        
        validation_rmse_history.append(avg_val_rmse)
        validation_edge_history.append(avg_val_edge)
        
        print(f"📊 Validation Epoch {epoch+1}: Mean RMSE={avg_val_rmse:.4f} | Mean Edge Err={avg_val_edge*100:.2f}%")            
    
    # ==========================================
    # Save epoch histories graphs
    # ==========================================
    
    epoch_history_path = os.path.join(run_dir, "training_history.png")
    plt.figure(figsize=(12, 10))
    
    epochs = np.arange(1, cfg.EPOCHS + 1)
    
    # Train Loss and Components
    plt.subplot(2, 1, 1)
    plt.plot(epochs, total_lost_history, label='Train Loss')
    plt.plot(epochs, rmse_history, label=f"RMSE x {cfg.LAMBDA_RMSE}")
    
    plt.plot(epochs, chamfer_loss_history, label=f"Chamfer Loss x {cfg.LAMBDA_CHAMFER}")
    plt.plot(epochs, edge_history, label=f"Edge Loss x {cfg.LAMBDA_EDGE}")
    plt.plot(epochs, area_history, label=f"Area Loss x {cfg.LAMBDA_AREA}")
    plt.plot(epochs, bend_history, label=f"Bend Loss x {cfg.LAMBDA_BEND}")
    plt.plot(epochs, pin_history, label=f"Pin Loss x {cfg.LAMBDA_PIN}")
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation History
    plt.subplot(2, 1, 2)
    plt.plot(epochs, validation_rmse_history, label='Validation RMSE')
    plt.plot(epochs, validation_edge_history, label='Validation Edge Error')
    plt.title('Validation History')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(epoch_history_path)
    plt.close()
    print(f"📈 Training history saved to: {epoch_history_path}")

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
    
    test_rmse_history = []
    test_edge_error_history = []
    time_per_frame = []

    for run_idx in unique_test_runs:
        validate_rollout(
            dataset=test_set, 
            model_ver=model_ver, 
            run_index=run_idx
        )
        avg_rmse_per_run, avg_edge_err_per_run, avg_time_per_frame = validate_metrics(
            dataset=test_set,
            model_ver=model_ver,
            run_index=run_idx
        )
        test_rmse_history.append(avg_rmse_per_run)
        test_edge_error_history.append(avg_edge_err_per_run)
        time_per_frame.append(avg_time_per_frame)
    
    # Save Training Details
    details = generate_details(
        train_loss=best_loss,
        test_rmse=np.mean(test_rmse_history),
        test_edge_err=np.mean(test_edge_error_history),
        time_per_frame=np.mean(time_per_frame),
        trainable_params=total_params
    )
    details_path = os.path.join(run_dir, "summary.txt")
    with open(details_path, 'w') as f:
        f.write(details)
        f.close()
    print(f"📄 Training summary saved to: {details_path}")
    
    print(f"✅ Returned model with Best Train Loss: {best_loss:.5f}")
    
    return model