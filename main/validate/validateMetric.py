import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from dataset_helpers.dataset import FlagWindDataset
from models.load_model import load_model

# def integrate_semi_implicit_euler(pos, vel, accel, dt):
#     """Standard Physics Integration"""
#     new_vel = vel + accel * dt
#     new_pos = pos + new_vel * dt
#     return new_pos, new_vel
def integrate(pos, vel, accel, dt):
    """
    Matches the 'PhysicsLoss' training logic:
    P_{t+1} = P_t + V_t*dt + 0.5*A*dt^2
    """
    # 1. Update Position (using Old Velocity + 0.5 * Accel)
    # This matches the Taylor expansion used in your Loss Function
    new_pos = pos + (vel * dt) + (0.5 * accel * (dt ** 2))
    
    # 2. Update Velocity
    new_vel = vel + (accel * dt)
    
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

def validate_metrics(dataset, model_ver, run_index=0, sub_dir=None, model=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Starting Numerical Validation for Run {run_index+1}...")

    # 1. Setup Data
    gt_flags = dataset.data_flags[run_index] # (Frames, Nodes, 6)
    gt_winds = dataset.data_winds[run_index]
    
    total_frames = gt_flags.shape[0]
    num_nodes = gt_flags.shape[1]

    if model is None:
        # 2. Load Model
        model = load_model(device)
        model_path = os.path.join(cfg.DATASET_DIR, "models", model_ver, "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model.eval()

    # 3. Load Topology (For Edge Error)
    edge_index_np = np.load(cfg.TOPOLOGY_PATH)
    edge_index = torch.from_numpy(edge_index_np).long().to(device)

    # 4. Prepare Stats
    def to_tensor(val):
        if torch.is_tensor(val):
            return val.to(device=device, dtype=torch.float32)
        return torch.tensor(val, device=device, dtype=torch.float32)
    
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
    
    # LSTM
    hidden_state = None
    
    # Store metrics per frame
    rmse_history = []
    edge_error_history = []
    
    print("🚀 Calculating Metrics...")
    
    T = 0
    
    for t in range(total_frames - 1):
        
        t1 = time.time()
        
        # --- A. Inference ---
        curr_state = torch.cat([curr_pos, curr_vel], dim=1)
        norm_state = (curr_state - mean_flag) / (std_flag + 1e-8)
        
        curr_wind = torch.from_numpy(gt_winds[t]).float().to(device)
        wind_expanded = curr_wind.mean(dim=0).unsqueeze(0).repeat(num_nodes, 1)
        norm_wind = (wind_expanded - mean_wind) / (std_wind + 1e-8)
        
        with torch.no_grad():
            if 'LSTM' in cfg.MODEL:
                # 1. Reshape for LSTM: (Batch, Seq_Len=1, Features)
                # We treat every node as a batch item
                lstm_input_nodes = norm_state.unsqueeze(1) # (N, 1, F)
                lstm_input_wind  = norm_wind.unsqueeze(1)  # (N, 1, F)
                
                # 2. Forward Pass with State
                # Ensure your LSTM forward method accepts and returns hidden_state!
                # If your model.forward() doesn't support passing hidden state, 
                # you strictly CANNOT validate autoregressively.
                
                # Note: This assumes your FlagLSTM_CNN_Net forward returns (output, hidden)
                # If it only returns output, the LSTM is resetting every frame (BAD).
                pred_norm_acc, hidden_state = model(lstm_input_nodes, lstm_input_wind, hidden=hidden_state)
                
                # Remove sequence dim for integration: (N, 1, 3) -> (N, 3)
                pred_norm_acc = pred_norm_acc.squeeze(1)
            else:
                pred_norm_acc = model(norm_state, norm_wind, edge_index)
        
        pred_real_acc = pred_norm_acc * std_acc + mean_acc
        
        if cfg.TARGET_TYPE == "accelerations":
            next_pos, next_vel = integrate(
                curr_pos, curr_vel, pred_real_acc, cfg.DELTA_T
            )
        elif cfg.TARGET_TYPE == "displacements":
            disp = pred_real_acc
            next_pos = curr_pos + disp
            next_vel = disp / cfg.DELTA_T
            
        t2 = time.time()
        T += (t2 - t1)
        
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
    
    avg_time_per_frame = T / (total_frames - 1)

    # 7. Generate Plots
    save_dir = os.path.join(cfg.DATASET_DIR, "models", model_ver)
    
    if sub_dir:
        save_dir = os.path.join(save_dir, sub_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    avg_rmse = np.mean(rmse_history)
    avg_edge_err = np.mean(edge_error_history)
    
    plot_metrics(rmse_history, edge_error_history, avg_rmse, avg_edge_err, save_dir, run_index)
    print(f"✅ Plots saved to {save_dir}")

    return avg_rmse, avg_edge_err, avg_time_per_frame

def plot_metrics(rmse, edge_err, avg_rmse, avg_edge_err, save_dir, run_index):
    """Generates and saves the validation graphs"""
    time_steps = np.arange(len(rmse))
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: RMSE (Drift)
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, rmse, label='Position RMSE', color='red')
    plt.axhline(y=avg_rmse, color='black', linestyle='--', label=f'Average: {avg_rmse:.3f} m')
    plt.title(f'Rollout Error Accumulation (Run {run_index+1})')
    plt.xlabel('Frame')
    plt.ylabel('RMSE (meters)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Edge Constraints (Stiffness)
    plt.subplot(1, 2, 2)
    # Convert to percentage
    edge_err_pct = [e * 100 for e in edge_err]
    plt.plot(time_steps, edge_err_pct, label='Edge Strain Error', color='blue')
    plt.axhline(y=avg_edge_err*100, color='black', linestyle='--', label=f'Average: {avg_edge_err*100:.2f}%')
    plt.title('Physical Plausibility (Edge Strain)')
    plt.xlabel('Frame')
    plt.ylabel('Avg Edge Strain')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"metrics_run_{run_index+1}.png"))
    plt.close()

if __name__ == "__main__":
    # Test on a single run
    train, test = FlagWindDataset.load_and_split(train_ratio=cfg.TRAIN_RATIO)
    
    # Validate Run 0 from the TEST set
    # validate_metrics(dataset=train, model_ver="004", run_index=1, sub_dir="temp")
    
    total_rmse = 0
    total_edge_err = 0
    
    for run_idx in range(0, 20):
        rmse, edge_err, time = validate_metrics(dataset=test, model_ver="020", run_index=run_idx, sub_dir="temp")
        total_rmse += rmse
        total_edge_err += edge_err
    
    print(f"Average RMSE: {total_rmse / 20:.3f}")
    print(f"Average Edge Error: {total_edge_err / 20:.3f}%")