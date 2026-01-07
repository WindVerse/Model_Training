import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from dataset_helpers.dataset import FlagWindDataset

# Dynamic Model Import
if cfg.MODEL == 'GNN':
    from models.GNN import FlagGraphNet as ModelClass
else:
    raise ValueError(f"Unknown MODEL: {cfg.MODEL}")

def integrate_semi_implicit_euler(pos, vel, accel, dt):
    """Standard Physics Integration"""
    new_vel = vel + accel * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel

def calculate_edge_lengths(pos, edge_index):
    """
    Computes the length of every edge in the mesh.
    pos: (Nodes, 3)
    edge_index: (2, Edges)
    Returns: (Edges,) length values
    """
    row, col = edge_index
    # Vector difference between connected nodes
    vec = pos[row] - pos[col]
    # Euclidean distance
    return torch.norm(vec, dim=1)

def validate_metrics(dataset, model_ver, run_index=0, sub_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Starting Numerical Validation for Run {run_index}...")

    # 1. Setup Data
    gt_flags = dataset.data_flags[run_index] # (Frames, Nodes, 6)
    gt_winds = dataset.data_winds[run_index]
    
    total_frames = gt_flags.shape[0]
    num_nodes = gt_flags.shape[1]

    # 2. Load Model
    model = ModelClass(
        in_node_dim=cfg.NODE_DIM,
        in_wind_dim=cfg.WIND_DIM,
        in_edge_dim=cfg.EDGE_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NO_GNN_LAYERS
    ).to(device)
    
    model_path = os.path.join(cfg.DATASET_DIR, "models", model_ver, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Load Topology (For Edge Error)
    edge_index_np = np.load(cfg.TOPOLOGY_PATH)
    edge_index = torch.from_numpy(edge_index_np).long().to(device)

    # 4. Prepare Stats
    def to_tensor(val): return torch.tensor(val, device=device).float()
    
    mean_flag = to_tensor(dataset.stats['flag_mean']).view(1, -1)
    std_flag  = to_tensor(dataset.stats['flag_std']).view(1, -1)
    mean_wind = to_tensor(dataset.stats['wind_mean']).view(1, -1)
    std_wind  = to_tensor(dataset.stats['wind_std']).view(1, -1)
    mean_acc  = to_tensor(dataset.stats['target_mean']).view(1, -1)
    std_acc   = to_tensor(dataset.stats['target_std']).view(1, -1)

    # 5. Calculate Initial Rest Lengths (Frame 0 GT)
    # We use this to compare how much the cloth stretches artificially
    initial_pos = torch.from_numpy(gt_flags[0, :, :3]).float().to(device)
    rest_lengths = calculate_edge_lengths(initial_pos, edge_index)

    # 6. ROLLOUT LOOP
    curr_pos = initial_pos.clone()
    curr_vel = torch.from_numpy(gt_flags[0, :, 3:]).float().to(device)
    
    # Store metrics per frame
    rmse_history = []
    edge_error_history = []
    
    print("🚀 Calculating Metrics...")
    
    for t in range(total_frames - 1):
        # --- A. Inference ---
        curr_state = torch.cat([curr_pos, curr_vel], dim=1)
        norm_state = (curr_state - mean_flag) / (std_flag + 1e-8)
        
        curr_wind = torch.from_numpy(gt_winds[t]).float().to(device)
        wind_expanded = curr_wind.mean(dim=0).unsqueeze(0).repeat(num_nodes, 1)
        norm_wind = (wind_expanded - mean_wind) / (std_wind + 1e-8)
        
        with torch.no_grad():
            pred_norm_acc = model(norm_state, norm_wind, edge_index)
        
        pred_real_acc = pred_norm_acc * std_acc + mean_acc
        
        next_pos, next_vel = integrate_semi_implicit_euler(
            curr_pos, curr_vel, pred_real_acc, cfg.DELTA_T
        )
        
        # --- B. Compute Metrics for this Frame ---
        
        # 1. Position RMSE (Drift)
        # Compare PREDICTED position vs GROUND TRUTH position at t+1
        gt_next_pos = torch.from_numpy(gt_flags[t+1, :, :3]).float().to(device)
        
        # Mean Squared Error for this frame
        mse = torch.mean((next_pos - gt_next_pos)**2)
        rmse = torch.sqrt(mse)
        rmse_history.append(rmse.item())
        
        # 2. Geometric Consistency (Stretching)
        # Calculate current edge lengths
        curr_lengths = calculate_edge_lengths(next_pos, edge_index)
        
        # Relative error: |Current - Rest| / Rest
        # We average this % error across all edges
        edge_err = torch.mean(torch.abs(curr_lengths - rest_lengths) / rest_lengths)
        edge_error_history.append(edge_err.item())

        # Update State
        curr_pos = next_pos
        curr_vel = next_vel

    # 7. Generate Plots
    save_dir = os.path.join(cfg.DATASET_DIR, "models", model_ver)
    
    if sub_dir:
        save_dir = os.path.join(save_dir, sub_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    plot_metrics(rmse_history, edge_error_history, save_dir, run_index)
    print(f"✅ Metrics saved to {save_dir}")

def plot_metrics(rmse, edge_err, save_dir, run_index):
    """Generates and saves the validation graphs"""
    time_steps = np.arange(len(rmse))
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: RMSE (Drift)
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, rmse, label='Position RMSE', color='red')
    plt.title(f'Rollout Error Accumulation (Run {run_index+1})')
    plt.xlabel('Frame')
    plt.ylabel('RMSE (meters)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Edge Constraints (Stiffness)
    plt.subplot(1, 2, 2)
    # Convert to percentage
    edge_err_pct = [e * 100 for e in edge_err]
    plt.plot(time_steps, edge_err_pct, label='Edge Stretch Error', color='blue')
    plt.title('Physical Plausibility (Stiffness)')
    plt.xlabel('Frame')
    plt.ylabel('Avg Edge Stretch (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"metrics_run_{run_index+1}.png"))
    plt.close()

if __name__ == "__main__":
    # Test on a single run
    train, test = FlagWindDataset.load_and_split(train_ratio=cfg.TRAIN_RATIO)
    
    # Validate Run 0 from the TEST set
    validate_metrics(dataset=train, model_ver="010", run_index=1, sub_dir="temp")