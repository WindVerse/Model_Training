import torch
import torch.nn as nn
import config as cfg

class FlagLSTM_CNN_Net(nn.Module):
    def __init__(self, 
                 in_node_dim=cfg.NODE_DIM, 
                 in_wind_dim=cfg.WIND_DIM, 
                 hidden_dim=cfg.HIDDEN_DIM, 
                 sequence_length=cfg.SEQUENCE_LENGTH,
                 num_lstm_layers=cfg.NO_LSTM_LAYERS,
                 cnn_channels=cfg.CNN_CHANNELS):
        super().__init__()

        self.grid_h = cfg.HEIGHT
        self.grid_w = cfg.WIDTH
        self.seq_len = sequence_length
        self.num_lstm_layers = num_lstm_layers

        # =========================================================
        # 1. DYNAMIC CNN ENCODER
        # =========================================================
        encoder_layers = []
        current_in_channels = in_node_dim
        
        # Loop through config list: e.g., [16, 32, 64]
        for out_channels in cnn_channels:
            encoder_layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=3, padding=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.MaxPool2d(2)) # Reduces size by half
            current_in_channels = out_channels # Update for next layer

        self.cnn_encoder = nn.Sequential(*encoder_layers)
        
        # ---------------------------------------------------------
        # Calculate Latent Size Dynamically
        # ---------------------------------------------------------
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_node_dim, self.grid_h, self.grid_w)
            dummy_out = self.cnn_encoder(dummy_input)
            
            self.flat_cnn_size = dummy_out.numel()
            self.latent_h = dummy_out.shape[2]
            self.latent_w = dummy_out.shape[3]
            self.last_channel_dim = dummy_out.shape[1] # e.g., 64

        self.fc_encode = nn.Linear(self.flat_cnn_size, hidden_dim)

        # =========================================================
        # 2. WIND ENCODER & LSTM
        # =========================================================
        self.wind_encoder = nn.Sequential(
            nn.Linear(in_wind_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.lstm_input_dim = hidden_dim + (hidden_dim // 4)
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # =========================================================
        # 3. DYNAMIC CNN DECODER
        # =========================================================
        # Project hidden state back to the shape of the Encoder's last feature map
        self.decoder_projection = nn.Linear(hidden_dim, self.last_channel_dim * self.latent_h * self.latent_w)

        decoder_layers = []
        
        # Reverse the channel list for decoding: e.g., [64, 32, 16]
        reversed_channels = cnn_channels[::-1]
        
        current_in_channels = reversed_channels[0] # Start with deepest (e.g., 64)

        # Loop to build the upsampling layers
        for i in range(len(reversed_channels) - 1):
            next_out_channels = reversed_channels[i+1]
            
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            decoder_layers.append(nn.Conv2d(current_in_channels, next_out_channels, kernel_size=3, padding=1))
            decoder_layers.append(nn.ReLU())
            
            current_in_channels = next_out_channels

        # Final Layer: Upsample to exact original Grid Size and output 3 channels (x,y,z)
        decoder_layers.append(nn.Upsample(size=(self.grid_h, self.grid_w), mode='bilinear', align_corners=False))
        decoder_layers.append(nn.Conv2d(current_in_channels, 3, kernel_size=3, padding=1))
        
        self.decoder_cnn = nn.Sequential(*decoder_layers)

    def forward(self, x_nodes, x_wind, edge_index=None, hidden=None):
        # --- HANDLE SEQUENCE INPUTS ---
        if x_nodes.dim() == 3:
            x_nodes = x_nodes.reshape(-1, x_nodes.shape[-1])
            
        if x_wind.dim() == 3:
            x_wind = x_wind.reshape(-1, x_wind.shape[-1])

        # 1. Handle Flattened Input
        total_points, D = x_nodes.shape 
        num_nodes = self.grid_h * self.grid_w
        
        B_total = total_points // num_nodes 

        # 2. Reshape & Encoder
        x_nodes_3d = x_nodes.reshape(B_total, num_nodes, D)
        x_wind_3d = x_wind.reshape(B_total, num_nodes, -1)
        
        # (Batch, C, H, W)
        x_img = x_nodes_3d.reshape(B_total, self.grid_h, self.grid_w, D).permute(0, 3, 1, 2) 

        cnn_feat = self.cnn_encoder(x_img)
        cnn_flat = cnn_feat.reshape(B_total, -1)
        flag_latent = self.fc_encode(cnn_flat)

        global_wind = x_wind_3d[:, 0, :] 
        wind_latent = self.wind_encoder(global_wind)

        # 3. LSTM Logic
        combined = torch.cat([flag_latent, wind_latent], dim=1) 
        
        # --- HIDDEN STATE LOGIC ---
        
        # ONNX uses model.eval(), so it will cleanly export the seq_len=1 branch.
        if not self.training:
            # Case A: Validation / Inference (Single Frame)
            lstm_in = combined.unsqueeze(1) # (Batch, 1, Features)
            
            if hidden is None:
                h0 = torch.zeros(self.num_lstm_layers, lstm_in.shape[0], self.lstm.hidden_size, device=x_nodes.device)
                c0 = torch.zeros(self.num_lstm_layers, lstm_in.shape[0], self.lstm.hidden_size, device=x_nodes.device)
                hidden = (h0, c0)

            lstm_out_seq, new_hidden = self.lstm(lstm_in, hidden)
            lstm_out_flat = lstm_out_seq[:, -1, :]
            target_batch_size = B_total 
            
        else:
            # Case B: Training (Full Sequences)
            true_batch_size = B_total // self.seq_len
            lstm_in = combined.reshape(true_batch_size, self.seq_len, -1)
            
            lstm_out, new_hidden = self.lstm(lstm_in) 
            lstm_out_flat = lstm_out.contiguous().reshape(B_total, -1)
            target_batch_size = B_total

        # 4. Decoder
        dec_in = self.decoder_projection(lstm_out_flat)
        dec_map = dec_in.reshape(target_batch_size, self.last_channel_dim, self.latent_h, self.latent_w)
        
        spatial_out = self.decoder_cnn(dec_map) 

        # 5. Output
        out_3d = spatial_out.permute(0, 2, 3, 1).contiguous().reshape(target_batch_size, num_nodes, 3)
        out_flat = out_3d.reshape(target_batch_size * num_nodes, 3)

        return out_flat, new_hidden