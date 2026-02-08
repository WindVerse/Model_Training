import torch
import torch.nn as nn
import config as cfg

class FlagLSTM_CNN_Net(nn.Module):
    def __init__(self, 
                 in_node_dim=cfg.NODE_DIM, 
                 in_wind_dim=cfg.WIND_DIM, # Typically 3
                 hidden_dim=cfg.HIDDEN_DIM, 
                 sequence_length=cfg.SEQUENCE_LENGTH):       # Set to >1 if your batching logic supports it
        super().__init__()

        self.grid_h = cfg.HEIGHT
        self.grid_w = cfg.WIDTH
        self.in_dim = in_node_dim
        self.seq_len = sequence_length

        # ---------------------------------------------------------
        # 1. CNN Encoder (Spatial Compression)
        # ---------------------------------------------------------
        # We use a simple 3-layer CNN to crush the 20x30 grid into a small latent vector
        self.cnn_encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_node_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by half
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces size by half
            
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduces size by half
        )
        
        # Calculate latent size dynamically based on grid size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_node_dim, self.grid_h, self.grid_w)
            dummy_out = self.cnn_encoder(dummy_input)
            self.flat_cnn_size = dummy_out.numel()
            self.latent_h = dummy_out.shape[2]
            self.latent_w = dummy_out.shape[3]

        self.fc_encode = nn.Linear(self.flat_cnn_size, hidden_dim)

        # ---------------------------------------------------------
        # 2. Wind Encoder (MLP)
        # ---------------------------------------------------------
        # Encodes the global wind vector (3D)
        self.wind_encoder = nn.Sequential(
            nn.Linear(in_wind_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # ---------------------------------------------------------
        # 3. Temporal Processing (LSTM)
        # ---------------------------------------------------------
        self.lstm_input_dim = hidden_dim + (hidden_dim // 4)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # ---------------------------------------------------------
        # 4. CNN Decoder (Spatial Reconstruction)
        # ---------------------------------------------------------
        # Project back to spatial latent map
        self.decoder_projection = nn.Linear(hidden_dim, 64 * self.latent_h * self.latent_w)

        # Use Upsample + Conv (Resize-Convolution) for robust output sizing
        self.decoder_cnn = nn.Sequential(
            # Upsample 1
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Upsample 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Upsample 3
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # Final sizing to exact HxW (handles odd/even mismatches from pooling)
            nn.Upsample(size=(self.grid_h, self.grid_w), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 3, kernel_size=3, padding=1) # Output: 3D acceleration
        )

    def forward(self, x_nodes, x_wind, edge_index=None):
        """
        Robust Forward Pass handling both Training (Sequence) and Inference (Single Frame)
        """
        # 1. Handle Flattened Input (Total_Nodes, Features)
        total_points, D = x_nodes.shape
        num_nodes = self.grid_h * self.grid_w
        
        # Calculate how many full "flags" we have in this batch
        # Training: 8000 nodes -> 40 flags (if seq=10, batch=4)
        # Validation: 600 nodes -> 1 flag
        B_total = total_points // num_nodes 

        # 2. Reshape back to 3D: (Batch_Total, Nodes, Features)
        x_nodes_3d = x_nodes.reshape(B_total, num_nodes, D)
        x_wind_3d = x_wind.reshape(B_total, num_nodes, -1)

        # 3. RESHAPE TO GRID (Recover Spatial Info)
        x_img = x_nodes_3d.reshape(B_total, self.grid_h, self.grid_w, D)
        x_img = x_img.permute(0, 3, 1, 2) 

        # 4. ENCODE SPATIAL
        cnn_feat = self.cnn_encoder(x_img)
        cnn_flat = cnn_feat.reshape(B_total, -1)
        flag_latent = self.fc_encode(cnn_flat)

        # 5. ENCODE WIND
        global_wind = x_wind_3d[:, 0, :] 
        wind_latent = self.wind_encoder(global_wind)

        # 6. TEMPORAL PROCESSING (LSTM) WITH DYNAMIC LOGIC
        combined = torch.cat([flag_latent, wind_latent], dim=1) # Shape: (B_total, Hidden)
        
        # --- 🛠️ FIX START ---
        if B_total < self.seq_len:
            # INFERENCE MODE (Single Frame or Small Batch)
            # If we don't have enough frames to make a sequence, we repeat the current frame.
            # Logic: "The state has been static for X frames"
            # Shape: (1, Hidden) -> (1, Seq_Len, Hidden)
            lstm_in = combined.unsqueeze(1).repeat(1, self.seq_len, 1)
            
            # The output will be (1, Seq_Len, Hidden), we just need the flattened version for the decoder
            # However, the decoder expects (B_total, ...) which is (1, ...).
            # The LSTM outputs a sequence. We take the LAST time step.
            lstm_out_seq, _ = self.lstm(lstm_in)
            lstm_out_flat = lstm_out_seq[:, -1, :] # Take last step: (1, Hidden)
            
            # Update B_total to match the decoder's expectation
            # (In inference, B_total is usually 1)
            target_batch_size = B_total 
            
        else:
            # TRAINING MODE (Full Sequences)
            # We assume B_total is a perfect multiple of seq_len
            true_batch_size = B_total // self.seq_len
            lstm_in = combined.reshape(true_batch_size, self.seq_len, -1)
            
            lstm_out, _ = self.lstm(lstm_in)
            
            # Flatten all time steps back because we predict loss for EVERY frame
            lstm_out_flat = lstm_out.contiguous().reshape(B_total, -1)
            target_batch_size = B_total
        # --- 🛠️ FIX END ---

        # 7. DECODE SPATIAL
        dec_in = self.decoder_projection(lstm_out_flat)
        dec_map = dec_in.reshape(target_batch_size, 64, self.latent_h, self.latent_w)
        spatial_out = self.decoder_cnn(dec_map) 

        # 8. FINAL OUTPUT
        out_3d = spatial_out.permute(0, 2, 3, 1).contiguous().reshape(target_batch_size, num_nodes, 3)
        out_flat = out_3d.reshape(target_batch_size * num_nodes, 3)

        return out_flat