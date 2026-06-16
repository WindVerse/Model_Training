import torch
import config as cfg

def apply_training_noise(flag_seq, target_seq, device):
    # flag_seq shape is typically [B, T, N, F]
    B, T, N, F = flag_seq.shape
    
    clean_curr = flag_seq[..., :3].clone()  # P_t
    clean_prev = flag_seq[..., 3:6].clone() # P_{t-1}
    
    # 1. Generate Raw Independent Noise
    noise_curr = torch.randn_like(clean_curr, device=device) * cfg.NOISE_STD
    noise_prev = torch.randn_like(clean_prev, device=device) * cfg.NOISE_STD
    
    # ==========================================================
    # THE FIX: SPATIAL NOISE ATTENUATION RAMP
    # ==========================================================
    # Create a ramp based on column index: Col 0 = 0.0, Col 1 = 0.33, Col 2 = 0.66, Col 3+ = 1.0
    cols = torch.arange(cfg.WIDTH, device=device).unsqueeze(0).repeat(cfg.HEIGHT, 1).view(-1, 1)
    ramp_mask = torch.clamp(cols.float() / 3.0, 0.0, 1.0) # Shape: [N, 1]
    
    # Reshape to broadcast over [B, T, N, 3]
    ramp_mask = ramp_mask.view(1, 1, N, 1).expand(-1, -1, -1, 3)
    
    # Apply the Ramp (Zeroes out the pole AND smooths the boundary shock)
    noise_curr = noise_curr * ramp_mask
    noise_prev = noise_prev * ramp_mask
    # ==========================================================
    
    flag_seq[..., :3] = clean_curr + noise_curr
    flag_seq[..., 3:6] = clean_prev + noise_prev
    
    # Exact Verlet mathematical target adjustment for two noisy frames:
    # A_adj = A_clean - 2(Noise_t) + Noise_{t-1}
    target_seq_adj = target_seq - (2.0 * noise_curr) + noise_prev
    
    return flag_seq, target_seq_adj

def flag_attack(model, x_nodes, x_wind, edge_index,
                target, curr_pos, curr_vel,
                criterion, optimizer):
    """
    Applies FLAG adversarial augmentation by iteratively updating a small
    perturbation on position and velocity features, while keeping pinned
    vertices unchanged across the entire flattened batch.
    """

    alpha = cfg.FLAG_STEP_SIZE
    M = cfg.FLAG_STEPS
    device = x_nodes.device

    # -----------------------------
    # 1. Initialize perturbation
    # -----------------------------
    delta = torch.zeros_like(x_nodes)

    # perturb positions and velocities
    delta[:, :6].uniform_(-alpha, alpha)

    # -----------------------------
    # 2. FIX BATCHING BUG & MASK PINNED VERTICES
    # -----------------------------
    # x_nodes is flattened: [B * T * N, Features]
    total_nodes = x_nodes.shape[0]
    N = cfg.PIN_MASK.shape[0]
    num_graphs_in_batch = total_nodes // N
    
    # Repeat the mask to cover the entire flattened batch
    batch_pin_mask = cfg.PIN_MASK.to(device).repeat(num_graphs_in_batch, 1) # [B*T*N, 1]
    
    # Invert it (0.0 for pinned, 1.0 for free)
    free_mask = 1.0 - batch_pin_mask
    
    # Zero out the initial perturbation for the pole
    delta = delta * free_mask

    delta = delta.to(device)
    delta.requires_grad_()

    optimizer.zero_grad()

    # -----------------------------
    # 3. FLAG adversarial steps
    # -----------------------------
    for step in range(M):

        out = model(x_nodes + delta, x_wind, edge_index)

        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out

        pred = pred.view_as(target)

        loss, rmse, chamfer, edge, area, bend, pin = criterion(
            pred, target, curr_pos, curr_vel
        )

        loss.backward(retain_graph=True)

        grad = delta.grad.detach()

        grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
        delta.data = delta.data + alpha * grad / grad_norm

        # Re-apply the mask to keep pinned nodes EXACTLY unchanged
        delta.data = delta.data * free_mask

        delta.grad.zero_()

    optimizer.step()

    return loss, rmse, chamfer, edge, area, bend, pin