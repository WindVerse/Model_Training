import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config as cfg

class PhysicsLoss(nn.Module):
    def __init__(self, 
                 initial_flag_pos,  # (N, 3) 
                 mean,              # Training set Mean
                 std,               # Training set Std
                 device="cuda"):
        super().__init__()

        # 1. Hyperparameters
        self.lambda_rmse = cfg.LAMBDA_RMSE
        self.lambda_positional = cfg.LAMBDA_POSITIONAL
        self.lambda_chamfer = cfg.LAMBDA_CHAMFER
        self.lambda_edge = cfg.LAMBDA_EDGE    # Penalize stretching
        self.lambda_area = cfg.LAMBDA_AREA    # Penalize shearing
        self.lambda_bend = cfg.LAMBDA_BEND    # Penalize bending
        self.lambda_pin = cfg.LAMBDA_PIN      # Penalize moving pinned nodes

        self.dt = cfg.DELTA_T

        # 2. Normalization Stats
        self.mean = torch.as_tensor(mean, device=device).view(1, 1, -1)
        self.std = torch.as_tensor(std, device=device).view(1, 1, -1)
        
        # 3. Load Topology (Edges & Faces)
        edge_index = np.load(cfg.TOPOLOGY_PATH)
        faces_np = np.load(cfg.FACES_PATH)
        
        self.src = torch.from_numpy(edge_index[0]).long().to(device)
        self.dst = torch.from_numpy(edge_index[1]).long().to(device)
        self.faces = torch.from_numpy(faces_np).long().to(device)

        # Build Adjacency Graph for Bending Loss (Pairs of faces sharing an edge)
        self.adj_faces = self._build_adjacent_faces(faces_np).to(device)

        # 4. Setup Rest State (Springs & Areas)
        initial_pos = torch.as_tensor(initial_flag_pos, device=device).float()
        pos_only = initial_pos[:, :3] 
        
        # Rest Edge Lengths
        rest_vec = pos_only[self.src] - pos_only[self.dst]
        self.rest_lengths = torch.norm(rest_vec, dim=1) # (Num_Edges,)

        # Rest Face Areas (Need dummy batch dim for the helper function)
        pos_batch = pos_only.unsqueeze(0) # (1, N, 3)
        rest_areas, _ = self.get_face_areas_and_normals(pos_batch)
        self.rest_areas = rest_areas.squeeze(0) # (Num_Faces,)

        # 5. DYNAMIC PINNED NODES (Using Config Mask)
        # Load the mask: (N, 1) where 1.0 = pinned, 0.0 = free
        pin_mask_tensor = cfg.PIN_MASK.to(device)
        # Extract the specific 1D indices of the pinned nodes for the pin_loss evaluation
        self.pinned_idx = torch.where(pin_mask_tensor.squeeze() == 1.0)[0]
        self.pinned_pos_target = pos_only[self.pinned_idx] # (N_Pin, 3)
        # Create an inverted mask for the integration step (0.0 for pinned, 1.0 for free)
        self.free_mask = 1.0 - pin_mask_tensor # Shape: (N, 1)

    def _build_adjacent_faces(self, faces_np):
        """Finds pairs of triangles that share an exact edge."""
        edges_to_faces = {}
        for face_idx, face in enumerate(faces_np):
            edges = [
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[1], face[2]))),
                tuple(sorted((face[2], face[0])))
            ]
            for edge in edges:
                if edge not in edges_to_faces:
                    edges_to_faces[edge] = []
                edges_to_faces[edge].append(face_idx)
        
        # Keep only interior edges shared by exactly 2 faces
        adj_pairs = [f_list for f_list in edges_to_faces.values() if len(f_list) == 2]
        return torch.tensor(adj_pairs, dtype=torch.long)

    def get_face_areas_and_normals(self, pos):
        """
        Calculates the area and the normal vector of each triangle face for a BATCH.
        pos: (B, N, 3)
        """
        # Gather the 3 corners of every triangle for the whole batch
        p0 = pos[:, self.faces[:, 0], :] # (B, Num_Faces, 3)
        p1 = pos[:, self.faces[:, 1], :]
        p2 = pos[:, self.faces[:, 2], :]

        v1 = p1 - p0
        v2 = p2 - p0

        # Cross Product along the coordinate dimension (dim=2)
        cross = torch.cross(v1, v2, dim=2)
        
        # Area is half the norm of the cross product
        areas = 0.5 * torch.norm(cross, dim=2) # (B, Num_Faces)
        
        # Normal vector is the normalized cross product
        normals = cross / (torch.norm(cross, dim=2, keepdim=True) + 1e-8) # (B, Num_Faces, 3)

        return areas, normals

    def de_normalize(self, tensor_norm):
        """Converts normalized predictions back into real-world physical acceleration."""
        return (tensor_norm * self.std) + self.mean

    def compute_chamfer_loss(self, pred_pos, target_pos):
        """Computes symmetric Chamfer distance between predicted and target point clouds."""
        dists = torch.cdist(pred_pos, target_pos) 
        min_dist_pred, _ = torch.min(dists, dim=2) 
        min_dist_target, _ = torch.min(dists, dim=1) 
        return torch.mean(min_dist_pred) + torch.mean(min_dist_target)
    
    def forward(self, pred_raw, target_raw, curr_pos, curr_vel):
        """
        Calculates the combined ML and Physics loss.
        pred_raw: Output from MeshGraphNet (Normalized Acceleration/Displacement)
        target_raw: True target (Raw Physical Space)
        curr_pos: Current position of nodes (Real Physical Space)
        curr_vel: Current velocity of nodes (Real Physical Space)
        """
        # # 1. Normalize ONLY the target (MeshGraphNet natively outputs in normalized space)
        # if cfg.MODEL == 'MeshGraphNet':   
        #     target_norm = (target_raw - self.mean) / (self.std + 1e-8)
        # else:
        #     target_norm = target_raw
            
        target_norm = (target_raw - self.mean) / (self.std + 1e-8)
        pred_norm = pred_raw

        # 2. Standard MSE Loss (Target matching in normalized space)
        mse_loss = F.mse_loss(pred_norm, target_norm)
        rmse_loss = torch.sqrt(mse_loss)

        # 3. INTEGRATION to physical space (MATCHING VALIDATION LOGIC)
        pred_real = self.de_normalize(pred_norm)
        target_real = self.de_normalize(target_norm)
        
        # ==========================================================
        # THE DYNAMIC BRICK WALL FIX
        # ==========================================================
        # Expand self.free_mask (N, 1) -> (1, N, 3) to broadcast over the batch
        free_mask_expanded = self.free_mask.unsqueeze(0).expand(-1, -1, 3)
        
        # Automatically zeroes out predictions for whatever nodes are marked in PIN_MASK
        pred_real_masked = pred_real * free_mask_expanded
        target_real_masked = target_real * free_mask_expanded
        # ==========================================================
        
        # Derive prev_pos from displacement (curr_vel passed from train_loop is actually curr_pos - prev_pos)
        # Since curr_vel = (curr_pos - prev_pos)  -->  prev_pos = curr_pos - curr_vel
        prev_pos = curr_pos - curr_vel

        if cfg.TARGET_TYPE in ["accelerations", "acc_new"]:
            # Semi-Implicit Euler
            pred_pos_next = curr_pos + curr_vel + (0.5 * pred_real_masked * (self.dt ** 2))
            target_pos_next = curr_pos + curr_vel + (0.5 * target_real_masked * (self.dt ** 2))
            
        elif cfg.TARGET_TYPE == "acc":
            # Verlet Integration
            pred_pos_next = (2 * curr_pos) - prev_pos + pred_real_masked
            target_pos_next = (2 * curr_pos) - prev_pos + target_real_masked
            
        elif cfg.TARGET_TYPE == "displacements":
            # Direct Geometric Addition
            pred_pos_next = curr_pos + pred_real_masked
            target_pos_next = curr_pos + target_real_masked
            
        else:
            raise ValueError(f"Unknown TARGET_TYPE: {cfg.TARGET_TYPE}")

        # 4. Positional & Chamfer Loss (Node-to-Node matching)
        positional_loss = torch.sqrt(F.mse_loss(pred_pos_next, target_pos_next))
        chamfer_loss = self.compute_chamfer_loss(pred_pos_next, target_pos_next)

        # 5. EDGE LOSS (Stretch Constraint)
        # Note: Updated to use Strain Percentage to prevent gradients from vanishing!
        p_src = pred_pos_next[:, self.src, :] # (B, E, 3)
        p_dst = pred_pos_next[:, self.dst, :] # (B, E, 3)
        curr_vec = p_src - p_dst
        curr_lengths = torch.norm(curr_vec, dim=2) # (B, E)
        
        # Use percentage strain instead of raw meters for stable gradients
        rest_lengths_expanded = self.rest_lengths.unsqueeze(0)
        strain = torch.abs(curr_lengths - rest_lengths_expanded) / (rest_lengths_expanded + 1e-8)
        edge_loss = torch.mean(strain ** 2)

        # 6. PIN LOSS (Anchor Constraint)
        current_pinned_pos = pred_pos_next[:, self.pinned_idx, :] # (B, N_Pin, 3)
        target_pos_expanded = self.pinned_pos_target.unsqueeze(0).expand_as(current_pinned_pos)
        pin_loss = F.mse_loss(current_pinned_pos, target_pos_expanded)

        # 7. SURFACE PHYSICS LOSSES
        pred_areas, pred_normals = self.get_face_areas_and_normals(pred_pos_next)

        # AREA LOSS (Shear Constraint)
        rest_areas_expanded = self.rest_areas.unsqueeze(0).expand_as(pred_areas)
        area_strain = torch.abs(pred_areas - rest_areas_expanded) / (rest_areas_expanded + 1e-8)
        area_loss = torch.mean(area_strain ** 2)

        # BENDING LOSS (Dihedral Angle Constraint)
        n1 = pred_normals[:, self.adj_faces[:, 0], :] # (B, Num_Adj_Pairs, 3)
        n2 = pred_normals[:, self.adj_faces[:, 1], :] 
        dot_product = torch.sum(n1 * n2, dim=2) # (B, Num_Adj_Pairs)
        bend_loss = torch.mean((1.0 - dot_product) ** 2)

        # 8. Total Loss Integration
        total_loss = (self.lambda_rmse * rmse_loss) + \
                     (self.lambda_positional * positional_loss) + \
                     (self.lambda_chamfer * chamfer_loss) + \
                     (self.lambda_edge * edge_loss) + \
                     (self.lambda_area * area_loss) + \
                     (self.lambda_bend * bend_loss) + \
                     (self.lambda_pin * pin_loss)

        return total_loss, rmse_loss, positional_loss, chamfer_loss, edge_loss, area_loss, bend_loss, pin_loss