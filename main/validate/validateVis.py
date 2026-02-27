import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from dataset_helpers.dataset import FlagWindDataset
from models.load_model import load_model

# def integrate_semi_implicit_euler(pos, vel, accel, dt):
#     """
#     Standard Physics Integration without manual constraints.
#     v_{t+1} = v_t + a * dt
#     p_{t+1} = p_t + v_{t+1} * dt
#     """
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

def validate_rollout(dataset, model_ver, run_index=0, sub_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎥 Starting Validation Rollout for Run {run_index+1} on {device}...")
    
    # Extract Ground Truth
    gt_flags = dataset.data_flags[run_index] 
    gt_winds = dataset.data_winds[run_index]
    
    total_frames = gt_flags.shape[0]
    num_nodes = gt_flags.shape[1]
    
    # 2. Load Model
    model = load_model(device)
    
    # Load Best Weights
    model_path = os.path.join(cfg.DATASET_DIR, "models", model_ver, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ Model Loaded.")

    # 3. Load Topology
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

    # 5. ROLLOUT LOOP
    print("🚀 Simulating...")
    
    # Initial State (Frame 0)
    curr_pos = torch.from_numpy(gt_flags[0, :, :3]).float().to(device)
    curr_vel = torch.from_numpy(gt_flags[0, :, 3:]).float().to(device)
    
    # LSTM
    hidden_state = None
    
    predictions = []
    
    for t in range(total_frames - 1):
        # A. Prepare Input
        # Normalize Current State
        curr_state = torch.cat([curr_pos, curr_vel], dim=1)
        norm_state = (curr_state - mean_flag) / (std_flag + 1e-8)
        
        # Prepare Wind
        curr_wind = torch.from_numpy(gt_winds[t]).float().to(device)
        wind_mean = curr_wind.mean(dim=0)
        wind_expanded = wind_mean.unsqueeze(0).repeat(num_nodes, 1)
        norm_wind = (wind_expanded - mean_wind) / (std_wind + 1e-8)
        
        # B. Model Inference
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
        
        # C. Physics Integration
        # De-normalize Acceleration
        pred_real_acc = pred_norm_acc * std_acc + mean_acc
        
        if cfg.TARGET_TYPE == "accelerations":
            next_pos, next_vel = integrate(
                curr_pos, curr_vel, pred_real_acc, cfg.DELTA_T
            )
        elif cfg.TARGET_TYPE == "displacements":
            disp = pred_real_acc
            next_pos = curr_pos + disp
            next_vel = disp / cfg.DELTA_T
        else:
            raise ValueError(f"Unknown TARGET_TYPE: {cfg.TARGET_TYPE}")
        
        # Store for visualization
        predictions.append(curr_pos.cpu().numpy())
        
        # Update
        curr_pos = next_pos
        curr_vel = next_vel

    # Add last frame
    predictions.append(curr_pos.cpu().numpy())
    predictions = np.array(predictions)
    
    print("✅ Rollout Complete. Generating Animation...")
    create_comparison_video(gt_flags[:, :, :3], predictions, model_ver, run_index, sub_dir=sub_dir)

def create_comparison_video(ground_truth, prediction, model_ver, run_index, sub_dir=None):
    """Creates a side-by-side 3D animation."""
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    ax1.set_title(f"Ground Truth (Run {run_index+1})")
    ax2.set_title("GNN Prediction (Rollout)")

    def setup_ax(ax):
        # Adjust these limits based on your actual data scale if needed
        ax.set_xlim(-1, 1) 
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    setup_ax(ax1)
    setup_ax(ax2)

    scat1 = ax1.scatter([], [], [], c='b', s=2)
    scat2 = ax2.scatter([], [], [], c='r', s=2)
    txt = fig.suptitle('')

    def update(frame):
        gt_p = ground_truth[frame]
        scat1._offsets3d = (gt_p[:,0], gt_p[:,1], gt_p[:,2])
        
        pred_p = prediction[frame]
        scat2._offsets3d = (pred_p[:,0], pred_p[:,1], pred_p[:,2])
        
        txt.set_text(f"Frame: {frame}/{len(ground_truth)}")
        return scat1, scat2

    ani = animation.FuncAnimation(fig, update, frames=len(ground_truth), interval=1000*cfg.DELTA_T, blit=False)
    
    # Save
    save_dir = os.path.join(cfg.DATASET_DIR, "models", model_ver)
    
    if sub_dir:
        save_dir = os.path.join(save_dir, sub_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"validation_run_{run_index+1}.mp4")
    
    try:
        ani.save(save_path, writer='ffmpeg', fps=20)
        print(f"🎬 Video saved to: {save_path}")
    except:
        print("⚠️ FFmpeg not found. Saving as GIF instead.")
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=20)
        print(f"🎬 GIF saved to: {save_path.replace('.mp4', '.gif')}")

if __name__ == "__main__":    
    train, test = FlagWindDataset.load_and_split(train_ratio=cfg.TRAIN_RATIO) 
    for run_idx in range(0, 20):
        validate_rollout(dataset=test, model_ver="114", run_index=run_idx, sub_dir="temp")