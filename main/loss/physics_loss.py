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

        # 5. Setup Pinned Nodes (Column 0)
        H, W = cfg.HEIGHT, cfg.WIDTH
        pinned_indices = [r * W for r in range(H)]
        self.pinned_idx = torch.tensor(pinned_indices, dtype=torch.long, device=device)
        self.pinned_pos_target = pos_only[self.pinned_idx] # (N_Pin, 3)

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
        return (tensor_norm * self.std) + self.mean

    def compute_chamfer_loss(self, pred_pos, target_pos):
        dists = torch.cdist(pred_pos, target_pos) 
        min_dist_pred, _ = torch.min(dists, dim=2) 
        min_dist_target, _ = torch.min(dists, dim=1) 
        return torch.mean(min_dist_pred) + torch.mean(min_dist_target)
    
    def forward(self, pred_norm, target_norm, curr_pos, curr_vel):
        # 1. Standard MSE Loss (Supervised)
        mse_loss = F.mse_loss(pred_norm, target_norm)
        rmse_loss = torch.sqrt(mse_loss)

        # 2. INTEGRATION
        pred_accel_real = self.de_normalize(pred_norm)
        pred_pos_next = curr_pos + (curr_vel * self.dt) + (0.5 * pred_accel_real * (self.dt ** 2))
        
        target_accel_real = self.de_normalize(target_norm)
        target_pos_next = curr_pos + (curr_vel * self.dt) + (0.5 * target_accel_real * (self.dt ** 2))
        
        # 3. Chamfer Loss
        chamfer_loss = self.compute_chamfer_loss(pred_pos_next, target_pos_next)

        # 4. EDGE LOSS (Stretch Constraint)
        p_src = pred_pos_next[:, self.src, :] # (B, E, 3)
        p_dst = pred_pos_next[:, self.dst, :] # (B, E, 3)
        curr_vec = p_src - p_dst
        curr_lengths = torch.norm(curr_vec, dim=2) # (B, E)
        
        length_diff = curr_lengths - self.rest_lengths.unsqueeze(0) # Broadcast to batch
        edge_loss = torch.mean(length_diff ** 2)

        # 5. PIN LOSS (Anchor Constraint)
        current_pinned_pos = pred_pos_next[:, self.pinned_idx, :] # (B, N_Pin, 3)
        target_pos_expanded = self.pinned_pos_target.unsqueeze(0).expand_as(current_pinned_pos)
        pin_loss = F.mse_loss(current_pinned_pos, target_pos_expanded)

        # ====================================================
        # SURFACE PHYSICS LOSSES
        # ====================================================
        pred_areas, pred_normals = self.get_face_areas_and_normals(pred_pos_next)

        # 6. AREA LOSS (Shear Constraint)
        # Prevents stretching diagonally. Compare predicted areas to rest areas.
        rest_areas_expanded = self.rest_areas.unsqueeze(0).expand_as(pred_areas)
        area_loss = F.mse_loss(pred_areas, rest_areas_expanded)

        # 7. BENDING LOSS (Dihedral Angle Constraint)
        # Prevents sharp folding. Computes dot product of normals of adjacent faces.
        n1 = pred_normals[:, self.adj_faces[:, 0], :] # (B, Num_Adj_Pairs, 3)
        n2 = pred_normals[:, self.adj_faces[:, 1], :] 
        
        # Dot product (sum along coordinate axis)
        dot_product = torch.sum(n1 * n2, dim=2) # (B, Num_Adj_Pairs)
        
        # Target is 1.0 (faces are perfectly flat/parallel to each other)
        bend_loss = torch.mean((1.0 - dot_product) ** 2)
        # ====================================================

        # 8. Total Loss Integration
        total_loss = (self.lambda_rmse * rmse_loss) + \
                     (self.lambda_chamfer * chamfer_loss) + \
                     (self.lambda_edge * edge_loss) + \
                     (self.lambda_area * area_loss) + \
                     (self.lambda_bend * bend_loss) + \
                     (self.lambda_pin * pin_loss)

        # Return the new losses as well so you can log them
        return total_loss, rmse_loss, chamfer_loss, edge_loss, area_loss, bend_loss, pin_loss