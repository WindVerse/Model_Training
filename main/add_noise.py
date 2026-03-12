import torch
import config as cfg

def apply_training_noise(flag_seq, target_seq, stats, device):
    """
    Injects random-walk noise into the inputs and adjusts the targets,
    teaching the model to correct for error accumulation (drift).
    """
    dt = cfg.DELTA_T
    
    # 1. Split pos and vel
    clean_pos = flag_seq[..., :3]
    clean_vel = flag_seq[..., 3:6]
    
    # 2. Generate Random Walk Noise
    # DeepMind applies Gaussian noise to velocity, and integrates it to position
    noise_v = torch.randn_like(clean_vel) * cfg.NOISE_STD
    noise_p = noise_v * dt 
    
    # 3. Mask out Pinned Nodes 
    H, W = cfg.HEIGHT, cfg.WIDTH
    pinned_indices = [r * W for r in range(H)]
    
    noise_v[:, :, pinned_indices, :] = 0.0
    noise_p[:, :, pinned_indices, :] = 0.0
    
    # 4. Apply noise to inputs
    noisy_pos = clean_pos + noise_p
    noisy_vel = clean_vel + noise_v
    noisy_flag_seq = torch.cat([noisy_pos, noisy_vel], dim=-1)
    
    # ==========================================
    # 5. ADJUST TARGET ACCELERATION
    # ==========================================
    # We want the model to predict an acceleration that forces the NOISY state
    # back to the perfect CLEAN next state.
    # Math for Taylor Integration: A_adj = A_clean - (4 * noise_v / dt)
    
    # A. Get normalization stats
    # Reshape to (1, 1, 1, 3) to broadcast across (B, T, N, 3)
    mean = torch.tensor(stats['target_mean'], device=device, dtype=torch.float32).view(1, 1, 1, 3)
    std = torch.tensor(stats['target_std'], device=device, dtype=torch.float32).view(1, 1, 1, 3)
    
    # B. De-normalize to real units (meters/s^2)
    target_real = (target_seq * std) + mean
    
    # C. Apply the physical shift
    target_real_adj = target_real - (4.0 * noise_v / dt)
    
    # D. Re-normalize so the MSE loss works correctly
    target_seq_adj = (target_real_adj - mean) / std
    
    return noisy_flag_seq, target_seq_adj

def flag_attack(model, x_nodes, x_wind, edge_index,
                target, curr_pos, curr_vel,
                criterion, optimizer):
    """
    Applies FLAG adversarial augmentation by iteratively updating a small
    perturbation on position and velocity features, while keeping pinned
    vertices unchanged.
    """

    alpha = cfg.FLAG_STEP_SIZE
    M = cfg.FLAG_STEPS

    device = x_nodes.device
    N = cfg.HEIGHT * cfg.WIDTH

    # -----------------------------
    # 1. Initialize perturbation
    # -----------------------------
    delta = torch.zeros_like(x_nodes)

    # perturb positions and velocities
    delta[:, :6].uniform_(-alpha, alpha)

    # -----------------------------
    # 2. Mask pinned vertices
    # -----------------------------
    H, W = cfg.HEIGHT, cfg.WIDTH
    pinned_indices = [r * W for r in range(H)]

    delta[pinned_indices, :] = 0.0

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

        # keep pinned nodes unchanged
        delta.data[pinned_indices, :] = 0.0

        delta.grad.zero_()

    optimizer.step()

    return loss, rmse, chamfer, edge, area, bend, pin