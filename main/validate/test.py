import importlib

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from dataset_helpers.dataset import FlagWindDataset
from models.load_model import load_model

def integrate(pos, vel, accel, dt):
    """
    P_{t+1} = P_t + V_t*dt + 0.5*A*dt^2
    """
    new_pos = pos + (vel * dt) + (0.5 * accel * (dt ** 2))
    new_vel = vel + (accel * dt)
    return new_pos, new_vel

def calculate_edge_lengths(pos, edge_index):
    """Computes the length of every edge in the mesh."""
    row, col = edge_index
    vec = pos[row] - pos[col]
    return torch.norm(vec, dim=1)

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

def create_comparison_video(ground_truth, prediction, save_dir, run_index, winds=None):
    """Creates a side-by-side 3D animation."""
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # --- Mini Wind Visualization ---
    wind_ax = fig.add_axes([0.4, 0.75, 0.25, 0.25], projection='3d')
    wind_ax.axis('off') 
    wind_ax.set_xlim(-1, 1); wind_ax.set_ylim(-1, 1); wind_ax.set_zlim(-1, 1)
    
    wind_mag_text = fig.text(0.5, 0.82, '', ha='center', va='top', fontsize=8, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    
    ax1.set_title(f"Ground Truth (Run {run_index+1})")
    ax2.set_title("GNN Prediction (Rollout)")

    def setup_ax(ax):
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    setup_ax(ax1)
    setup_ax(ax2)

    scat1 = ax1.scatter([], [], [], c='b', s=2)
    scat2 = ax2.scatter([], [], [], c='r', s=2)
    txt = fig.suptitle('')
    
    wind_quivers = []

    def update(frame):
        gt_p = ground_truth[frame]
        scat1._offsets3d = (gt_p[:,0], gt_p[:,1], gt_p[:,2])
        
        pred_p = prediction[frame]
        scat2._offsets3d = (pred_p[:,0], pred_p[:,1], pred_p[:,2])
        
        txt.set_text(f"Frame: {frame}/{len(ground_truth)}")
        
        if winds is not None:
            nonlocal wind_quivers
            if wind_quivers:
                for quiv in wind_quivers:
                    quiv.remove()
            wind_quivers.clear()
            
            avg_w = np.mean(winds[frame], axis=0)
            mag = np.sqrt(avg_w[0]**2 + avg_w[1]**2 + avg_w[2]**2)
            
            if mag > 1e-8:
                dir_w = avg_w / mag
            else:
                dir_w = np.zeros(3)
                
            quiv = wind_ax.quiver(0, 0, 0, dir_w[0], dir_w[1], dir_w[2], 
                                  length=1.0, color='magenta', linewidth=3, arrow_length_ratio=0.3)
            wind_quivers.append(quiv)
            wind_mag_text.set_text(f"Wind Strength: {mag:.4f}")
            
        return scat1, scat2, txt, wind_mag_text
        
    ani = animation.FuncAnimation(fig, update, frames=len(ground_truth), interval=1000*cfg.DELTA_T, blit=False)
    
    save_path = os.path.join(save_dir, f"validation_run_{run_index+1}.mp4")
    try:
        ani.save(save_path, writer='ffmpeg', fps=cfg.FPS)
        print(f"Video saved to: {save_path}")
    except:
        print("FFmpeg not found. Saving as GIF instead.")
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=cfg.FPS)
        print(f"GIF saved to: {save_path.replace('.mp4', '.gif')}")

def create_prediction_video(prediction, save_dir, run_index, winds=None):
    """Creates a 3D animation for the GNN prediction only."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- Mini Wind Visualization ---
    wind_ax = fig.add_axes([0.65, 0.75, 0.25, 0.25], projection='3d')
    wind_ax.axis('off') 
    wind_ax.set_xlim(-1, 1); wind_ax.set_ylim(-1, 1); wind_ax.set_zlim(-1, 1)
    
    wind_mag_text = fig.text(0.78, 0.82, '', ha='center', va='top', fontsize=8, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    
    ax.set_title("GNN Prediction (Rollout)")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    scat = ax.scatter([], [], [], c='r', s=2)
    txt = fig.suptitle('')
    
    wind_quivers = []

    def update(frame):
        pred_p = prediction[frame]
        scat._offsets3d = (pred_p[:,0], pred_p[:,1], pred_p[:,2])
        txt.set_text(f"Frame: {frame}/{len(prediction)}")
        
        if winds is not None:
            nonlocal wind_quivers
            if wind_quivers:
                for quiv in wind_quivers:
                    quiv.remove()
            wind_quivers.clear()
            
            avg_w = np.mean(winds[frame], axis=0)
            mag = np.linalg.norm(avg_w)
            
            dir_w = avg_w / mag if mag > 1e-8 else np.zeros(3)
                
            quiv = wind_ax.quiver(0, 0, 0, dir_w[0], dir_w[1], dir_w[2], 
                                  length=1.0, color='magenta', linewidth=3, arrow_length_ratio=0.3)
            wind_quivers.append(quiv)
            wind_mag_text.set_text(f"Wind: {mag:.4f}")
            
        return scat, txt, wind_mag_text
        
    ani = animation.FuncAnimation(fig, update, frames=len(prediction), interval=1000*cfg.DELTA_T, blit=False)
    
    save_path = os.path.join(save_dir, f"prediction_run_{run_index+1}.mp4")
    try:
        ani.save(save_path, writer='ffmpeg', fps=cfg.FPS)
        print(f"Video saved to: {save_path}")
    except:
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=cfg.FPS)
        print(f"GIF saved to: {save_path.replace('.mp4', '.gif')}")

def test_run(dataset, target, num_of_digits, run_index=0, model=None):
    """Runs inference ONCE, calculates metrics, and generates animation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nStarting Historical Validation for Run {run_index+1} on {device}...")

    # 1. Setup Data
    gt_flags = dataset.data_flags[run_index]
    gt_winds = dataset.data_winds[run_index]
    
    total_frames = gt_flags.shape[0]
    num_nodes = gt_flags.shape[1]

    # 2. Load Model
    if model is None:
        model = load_model(device)
        model_path = os.path.join(cfg.DATASET_DIR, "models", str(model_ver), "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model.eval()

    # 3. Load Topology & Stats
    edge_index = torch.from_numpy(np.load(cfg.TOPOLOGY_PATH)).long().to(device)

    def to_tensor(val):
        if torch.is_tensor(val): return val.to(device=device, dtype=torch.float32)
        return torch.tensor(val, device=device, dtype=torch.float32)
    
    mean_flag = to_tensor(dataset.stats['flag_mean']).view(1, -1)
    std_flag  = to_tensor(dataset.stats['flag_std']).view(1, -1)
    mean_wind = to_tensor(dataset.stats['wind_mean']).view(1, -1)
    std_wind  = to_tensor(dataset.stats['wind_std']).view(1, -1)
    mean_acc  = to_tensor(dataset.stats['target_mean']).view(1, -1)
    std_acc   = to_tensor(dataset.stats['target_std']).view(1, -1)

    # 4. Initialize Historical State Buffer
    initial_pos = torch.from_numpy(gt_flags[0, :, :3]).float().to(device)
    rest_lengths = calculate_edge_lengths(initial_pos, edge_index)

    # NEW: Create a rolling buffer filled with copies of the initial position
    history_buffer = deque([initial_pos.clone() for _ in range(cfg.HISTORY_WINDOW)], maxlen=cfg.HISTORY_WINDOW)
    
    hidden_state = None
    
    # 5. Tracking Variables
    predictions = [initial_pos.cpu().numpy()] # Store frame 0
    rmse_history = []
    edge_error_history = []
    
    T = 0
    
    # 6. UNIFIED INFERENCE LOOP
    print("Simulating and Computing Metrics...")
    for t in range(total_frames - 1):
        t1 = time.time()
        
        # --- A. Extract Current and Previous Positions ---
        curr_pos = history_buffer[-1]
        prev_pos = history_buffer[-2]
        
        # ==========================================
        # 1. GLOBAL SCALING (Velocity & Wind)
        # ==========================================
        # Calculate Kinematic Velocity
        curr_vel = (curr_pos - prev_pos)
        curr_vel_scaled = curr_vel * cfg.VEL_UP  # Scale up globally
        
        # Get Wind and Expand Spatially
        curr_wind_raw = torch.from_numpy(gt_winds[t]).float().to(device) # Shape: (8, 3)
        
        x, y, z = curr_pos[:, 0], curr_pos[:, 1], curr_pos[:, 2]
        ix, iy, iz = (x >= 0).long(), (y >= 0).long(), (z >= 0).long()
        cube_index = ix*4 + iy*2 + iz
        cube_index_expanded = cube_index.unsqueeze(-1).expand(-1, 3)
        
        wind_expanded = torch.gather(curr_wind_raw, 0, cube_index_expanded)
        wind_expanded_scaled = wind_expanded / cfg.WIND_DOWN  # Scale down globally
        
        # ==========================================
        # 2. BUILD MODEL-SPECIFIC INPUTS
        # ==========================================
        if cfg.MODEL == 'MeshGraphNet':
            # 3. BUILD NODE FEATURES (Scaled Vel + Scaled Wind + Pin Mask)
            batch_pin_mask = cfg.PIN_MASK.to(device)
            node_features = torch.cat([curr_vel_scaled, wind_expanded_scaled, batch_pin_mask], dim=-1)

            # 4. BUILD EDGE FEATURES 
            row, col = edge_index
            x_ij = curr_pos[row] - curr_pos[col]
            x_ij_norm = torch.norm(x_ij, p=2, dim=-1, keepdim=True)
            v_ij = curr_vel_scaled[row] - curr_vel_scaled[col] # Uses scaled relative velocity
            rest_lengths_expanded = rest_lengths.unsqueeze(-1)
            
            edge_attr = torch.cat([x_ij, x_ij_norm, v_ij, rest_lengths_expanded], dim=-1)
            
            # --- B. Inference ---
            with torch.no_grad():
                pred_norm_acc = model(node_features, edge_index, edge_attr)
                
        else:
            # A. Prepare Input for GNN / LSTM
            spatial_mean = mean_flag[0, :3].view(1, 3).to(device)
            spatial_std = std_flag[0, :3].view(1, 3).to(device)
            curr_pos_norm = (curr_pos - spatial_mean) / (spatial_std + 1e-8)
            
            curr_state = torch.cat([curr_pos_norm, curr_vel_scaled], dim=1) 
            
            # B. Model Inference
            with torch.no_grad():
                if 'LSTM' in cfg.MODEL:
                    lstm_input_nodes = curr_state.unsqueeze(1)
                    lstm_input_wind  = wind_expanded_scaled.unsqueeze(1) 
                    pred_norm_acc, hidden_state = model(lstm_input_nodes, lstm_input_wind, hidden=hidden_state)
                    pred_norm_acc = pred_norm_acc.squeeze(1)
                else:
                    pred_norm_acc = model(curr_state, wind_expanded_scaled, edge_index)
        
        
        # --- C. DENORMALIZE THE PREDICTION ---
        pred_real_acc = pred_norm_acc * std_acc + mean_acc
        
        # ==========================================
        # ENFORCE BOUNDARY CONDITIONS (Fixed Masking)
        # ==========================================
        # Use the global PIN_MASK so it automatically handles Row-Major vs Sequential!
        batch_pin_mask = cfg.PIN_MASK.to(device) # Shape: [N, 1]
        free_mask = 1.0 - batch_pin_mask
        
        # Smoothly zero out any rogue boundary accelerations
        pred_real_acc = pred_real_acc * free_mask
        
        # --- C. Physics Integration ---
        # Calculate instantaneous kinematic velocity from the two most recent frames in the buffer
        kinematic_vel = (history_buffer[-1] - history_buffer[-2]) / cfg.DELTA_T
        
        if cfg.TARGET_TYPE in ["accelerations", "acc_new"]:
            next_pos, _ = integrate(history_buffer[-1], kinematic_vel, pred_real_acc, cfg.DELTA_T)
        elif cfg.TARGET_TYPE == "acc":
            next_pos = (2 * curr_pos) - prev_pos + pred_real_acc
        elif cfg.TARGET_TYPE == "displacements":
            disp = pred_real_acc
            next_pos = history_buffer[-1] + disp
            
        t2 = time.time()
        T += (t2 - t1)
        
        # --- D. Compute Metrics ---
        gt_next_pos = torch.from_numpy(gt_flags[t+1, :, :3]).float().to(device)
        
        mse = torch.mean((next_pos - gt_next_pos)**2)
        rmse_history.append(torch.sqrt(mse).item())
        
        curr_lengths = calculate_edge_lengths(next_pos, edge_index)
        edge_err = torch.mean(torch.abs(curr_lengths - rest_lengths) / rest_lengths)
        edge_error_history.append(edge_err.item())

        # --- E. Update Tracking ---
        predictions.append(next_pos.cpu().numpy())
        
        # This automatically pushes the oldest frame out and adds the new position
        history_buffer.append(next_pos)
    
    # 7. Finalize & Save
    avg_time_per_frame = T / (total_frames - 1)
    avg_rmse = np.mean(rmse_history)
    avg_edge_err = np.mean(edge_error_history)
        
    save_dir = os.path.join(cfg.DATASET_DIR,"results")
        
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating Plots and Animation...")
    plot_metrics(rmse_history, edge_error_history, avg_rmse, avg_edge_err, save_dir, run_index)
    # create_comparison_video(gt_flags[:, :, :3], np.array(predictions), save_dir, run_index, winds=gt_winds)
    create_prediction_video(np.array(predictions), os.path.join(cfg.DATASET_DIR, "results"), run_index, winds=gt_winds)

    return avg_rmse, avg_edge_err, avg_time_per_frame

if __name__ == "__main__":
    custom_dataset_dir = "../../datasets/test_set"  
    target = "acc"
    num_of_digits = 4
    pin_mask_inversed = True

    # ==========================================
    # 1. DYNAMIC ARCHITECTURE HOT-SWAP
    # ==========================================
    # We must load the config_used BEFORE loading the dataset so that 
    # HISTORY_WINDOW, NODE_DIM, VEL_UP, etc., perfectly match the model!
    config_used_path = os.path.join(custom_dataset_dir, "config_used.py")
    if os.path.exists(config_used_path):
        print(f"Loading trained architecture config from: {config_used_path}")
        spec = importlib.util.spec_from_file_location("config_used", config_used_path)
        config_used = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_used)
        
        # Overwrite all uppercase constants EXCEPT paths
        for key in dir(config_used):
            if key.isupper() and not key.endswith("_DIR") and not key.endswith("_PATH") and key != "TARGET_TYPE":
                setattr(cfg, key, getattr(config_used, key))
    else:
        print(f"WARNING: {config_used_path} not found! Using default config.py architecture.")

    # ==========================================
    # 2. OVERRIDE DIRECTORIES FOR CUSTOM TEST RUN
    # ==========================================
    cfg.IS_TEST = False
    cfg.NO_DIGITS = num_of_digits
    cfg.TARGET_TYPE = target
    
    cfg.DATASET_DIR = custom_dataset_dir
    cfg.FLAG_DIR = os.path.join(custom_dataset_dir, "flags")
    cfg.WIND_DIR = os.path.join(custom_dataset_dir, "winds")
    cfg.TARGET_DIR = os.path.join(custom_dataset_dir, "targets", cfg.TARGET_TYPE)
    
    cfg.TOPOLOGY_PATH = os.path.join(custom_dataset_dir, "topology", "topology_edge_index.npy")
    cfg.FACES_PATH = os.path.join(custom_dataset_dir, "topology", "topology_faces.npy")
    cfg.ITERATION_COUNT = 1
    cfg.MAX_FRAMES = 1000
    
    # ---------------------------------------------------------
    # 🌟 THE FIX: DYNAMIC PIN MASK OVERRIDE FOR TEST DATASET 🌟
    # ---------------------------------------------------------
    # The new test dataset uses sequential pinning (0, 1, ..., H-1) 
    # instead of the old Row-Major pinning (0, W, 2W...)
    if pin_mask_inversed:
        H, W = cfg.HEIGHT, cfg.WIDTH
        new_pin_mask = torch.zeros((H * W, 1))
        
        for r in range(H):
            new_pin_mask[r, 0] = 1.0  # Sequential indexing constraint
            
        cfg.PIN_MASK = new_pin_mask

    # ==========================================
    # 3. LOAD DATASET (Now using correct paths and config!)
    # ==========================================
    print(f"Loading dataset from: {custom_dataset_dir}")
    test_dataset, _ = FlagWindDataset.load_and_split(train_ratio=1.0)
    
    # ==========================================
    # 4. INJECT TRAINED NORMALIZATION STATS
    # ==========================================
    stats_path = os.path.join(custom_dataset_dir, "train_stats.npz")
    if os.path.exists(stats_path):
        print(f"Injecting trained normalization stats from: {stats_path}")
        trained_stats = np.load(stats_path)
        test_dataset.stats = {
            'flag_mean': torch.from_numpy(trained_stats['flag_mean']),
            'flag_std': torch.from_numpy(trained_stats['flag_std']),
            'wind_mean': torch.from_numpy(trained_stats['wind_mean']),
            'wind_std': torch.from_numpy(trained_stats['wind_std']),
            'target_mean': torch.from_numpy(trained_stats['target_mean']),
            'target_std': torch.from_numpy(trained_stats['target_std'])
        }
    else:
        print(f"WARNING: {stats_path} not found! Physics integration will fail.")

    # ==========================================
    # 5. PRE-LOAD THE CUSTOM MODEL
    # ==========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(custom_dataset_dir, "best_model.pth")
    
    print(f"Loading custom model weights from: {model_path}")
    custom_model = load_model(device)
    custom_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    custom_model.eval()

    # ==========================================
    # 6. RUN VALIDATION
    # ==========================================
    total_rmse = 0
    total_edge_err = 0
    total_time = 0
    
    runs_to_test = len(test_dataset.data_flags)
    print(f"\nEvaluating {runs_to_test} run(s)...")
    
    for run_idx in range(runs_to_test):
        rmse, edge_err, time_pf = test_run(
            dataset=test_dataset, 
            target=target,
            num_of_digits=num_of_digits,
            run_index=run_idx, 
            model=custom_model
        )
        total_rmse += rmse
        total_edge_err += edge_err
        total_time += time_pf
    
    if runs_to_test > 0:
        print("\n=== Validation Complete ===")
        print(f"Average RMSE over {runs_to_test} runs: {total_rmse / runs_to_test:.3f} m")
        print(f"Average Edge Error over {runs_to_test} runs: {total_edge_err / runs_to_test * 100:.3f}%")
        print(f"Average Time per Frame: {total_time / runs_to_test:.3f} seconds")
    else:
        print("\nNo runs were found to test. Check your dataset folder paths and naming conventions!")