import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import copy

import config as cfg

class FlagWindDataset(Dataset):
    def __init__(self, data_flags, data_winds, data_targets, samples, sequence_length, stats=None):
        """
        Internal Init: Do not call directly. Use 'load_and_split' instead.
        """
        self.data_flags = data_flags   # Shared Reference (Low Memory)
        self.data_winds = data_winds   # Shared Reference
        self.data_targets = data_targets # Shared Reference
        self.samples = samples         # Unique to this split
        self.sequence_length = sequence_length
        self.stats = stats             # Shared Stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        run_idx, start_frame = self.samples[idx]
        end_frame = start_frame + self.sequence_length

        flag = self.data_flags[run_idx][start_frame:end_frame]
        wind = self.data_winds[run_idx][start_frame:end_frame]
        target = self.data_targets[run_idx][start_frame:end_frame]

        flag = torch.from_numpy(flag).float()
        wind = torch.from_numpy(wind).float()
        target = torch.from_numpy(target).float()

        # Normalize (using Pre-calculated Stats)
        if self.stats:
            flag = (flag - self.stats['flag_mean']) / (self.stats['flag_std'] + 1e-8)
            wind = (wind - self.stats['wind_mean']) / (self.stats['wind_std'] + 1e-8)
            target = (target - self.stats['target_mean']) / (self.stats['target_std'] + 1e-8)

        return flag, wind, target

    # ==========================================
    #  BUILDER METHOD
    # ==========================================
    @classmethod
    def load_and_split(cls, train_ratio=0.8, max_frames=cfg.MAX_FRAMES, sequence_length=cfg.SEQUENCE_LENGTH):
        print(f"Loading Dataset (Split Ratio: {train_ratio})...")
        
        # Load ALL Data into RAM
        all_flags, all_winds, all_targets = [], [], []
        
        run_metadata = [] 

        valid_frames = max_frames - 1 

        for iteration in tqdm(range(1, cfg.ITERATION_COUNT + 1), desc="Loading All Runs"):
            run_flags, run_winds, run_targets = [], [], []
            
            for frame in range(valid_frames):
                f_path = os.path.join(cfg.FLAG_DIR, f"flag_{iteration:03d}_{frame:03d}.npy")
                w_path = os.path.join(cfg.WIND_DIR, f"wind_{iteration:03d}_{frame:03d}.npy")
                t_path = os.path.join(cfg.TARGET_DIR, f"target_{iteration:03d}_{frame:03d}.npy")

                if not os.path.exists(t_path):
                    print(f"Warning: Missing target file {t_path}. Ending run load early.")
                    break
                
                run_flags.append(np.load(f_path))
                run_winds.append(np.load(w_path))
                run_targets.append(np.load(t_path))

            if len(run_flags) >= sequence_length:
                # Store Data
                all_flags.append(np.stack(run_flags))
                all_winds.append(np.stack(run_winds))
                all_targets.append(np.stack(run_targets))
                
                # Store Metadata
                run_idx = len(all_flags) - 1
                run_metadata.append((run_idx, iteration))

        print("Data Loaded. Splitting...")

        # Determine Split Boundary
        total_runs = len(run_metadata)
        num_train = int(total_runs * train_ratio)
        
        # Get the Iteration ID where training ends (e.g., Iteration 80)
        # run_metadata[i] is (index, iteration)
        train_limit_iter = run_metadata[num_train - 1][1] if num_train > 0 else 0
        
        print(f"Total Runs: {total_runs}. Train Count: {num_train}. Test Count: {total_runs - num_train}.")

        # Create Sample Indices
        train_samples = []
        test_samples = []

        # Logic: Iterate through loaded data metadata
        for run_idx, iteration in run_metadata:
            num_frames = len(all_flags[run_idx])
            valid_starts = num_frames - sequence_length + 1
            
            for start_t in range(valid_starts):
                sample = (run_idx, start_t)
                
                if iteration <= train_limit_iter:
                    train_samples.append(sample)
                else:
                    test_samples.append(sample)

        # Calculate Stats (ON TRAIN SET ONLY)
        print("Calculating Statistics on TRAIN set only...")
        
        # We need to gather all arrays that belong to the train set
        train_indices = [idx for idx, itr in run_metadata if itr <= train_limit_iter]
        
        def compute_stats(data_source, indices):
            # Select only training runs
            selected_data = [data_source[i] for i in indices]
            full_stack = np.concatenate(selected_data, axis=0)
            mean = np.mean(full_stack, axis=(0, 1), keepdims=True)
            std = np.std(full_stack, axis=(0, 1), keepdims=True)
            return torch.from_numpy(mean).float(), torch.from_numpy(std).float()

        stats = {}
        stats['flag_mean'], stats['flag_std'] = compute_stats(all_flags, train_indices)
        stats['wind_mean'], stats['wind_std'] = compute_stats(all_winds, train_indices)
        stats['target_mean'], stats['target_std'] = compute_stats(all_targets, train_indices)
        
        # Save Stats for Later Use
        stats_path = os.path.join(cfg.DATASET_DIR, f"stats{'Test' if cfg.IS_TEST else ''}.txt")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            f.write(str(stats))
            f.close()
        print(f"Statistics Calculated and Saved to {stats_path}.")

        # Create Dataset Objects
        
        train_ds = cls(all_flags, all_winds, all_targets, train_samples, sequence_length, stats)
        test_ds = cls(all_flags, all_winds, all_targets, test_samples, sequence_length, stats)

        print(f"✅ Splitting Complete.")
        print(f"Train Samples: {len(train_ds)}")
        print(f"Test Samples:  {len(test_ds)}")

        return train_ds, test_ds