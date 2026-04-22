import torch
import config as cfg

def apply_training_noise(flag_seq, target_seq, device):
    """
    Injects random-walk noise into the current position and adjusts the target acceleration 
    using MeshGraphNets Verlet integration math.
    """
    # 1. Extract Current Position (P_t) from the sequence
    clean_pos = flag_seq[..., :3].clone()
    
    # 2. Generate Random Walk Noise ONLY for position
    pos_noise = torch.randn_like(clean_pos, device=device) * cfg.NOISE_STD
    
    # 3. Mask out Pinned Nodes (so the pole doesn't shake)
    H, W = cfg.HEIGHT, cfg.WIDTH
    pinned_indices = [r * W for r in range(H)]
    pos_noise[:, :, pinned_indices, :] = 0.0
    
    # 4. Apply noise to the sequence
    flag_seq[..., :3] = clean_pos + pos_noise
    
    # 5. Verlet Target Adjustment: A_adj = A_clean - (2.0 * pos_noise)
    target_seq_adj = target_seq - (2.0 * pos_noise)
    
    return flag_seq, target_seq_adj

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