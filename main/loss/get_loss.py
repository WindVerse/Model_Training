import config as cfg

def getLoss(train_set, device):
    if cfg.LOSS == 'physicsLoss':
        from loss.physics_loss import PhysicsLoss
        initial_pos_ref = train_set.data_flags[0][0, :, :3] 
        criterion = PhysicsLoss(
            initial_flag_pos=initial_pos_ref,
            mean=train_set.stats['target_mean'],
            std=train_set.stats['target_std'],
            device=device
        )
        return criterion
    elif cfg.LOSS == 'L2Loss':
        from loss.l2_loss import L2Loss
        criterion = L2Loss(
            mean=train_set.stats['target_mean'],
            std=train_set.stats['target_std'],
            device=device
        )
        return criterion
    else:
        raise ValueError(f"Unknown LOSS in config: {cfg.LOSS}")