import os
import re
import subprocess
import itertools

# 1. Define your parameter grids
model_params = [
    # {"BATCH_SIZE": 4, "LEARNING_RATE": 0.0001, "NO_GNN_LAYERS": 5, "NO_MLP_HIDDEN_LAYERS": 5, "HIDDEN_DIM": 128},
    {"BATCH_SIZE": 8, "LEARNING_RATE": 0.0001, "NO_GNN_LAYERS": 5, "NO_MLP_HIDDEN_LAYERS": 5, "HIDDEN_DIM": 128},
    # {"BATCH_SIZE": 4, "LEARNING_RATE": 0.001, "NO_GNN_LAYERS": 5, "NO_MLP_HIDDEN_LAYERS": 5, "HIDDEN_DIM": 128},
    {"BATCH_SIZE": 8, "LEARNING_RATE": 0.001, "NO_GNN_LAYERS": 5, "NO_MLP_HIDDEN_LAYERS": 5, "HIDDEN_DIM": 128},
]

loss_params = [
    {"LAMBDA_RMSE": 1, "LAMBDA_POSITIONAL": 5, "LAMBDA_CHAMFER": 0, "LAMBDA_EDGE": 10, "LAMBDA_AREA": 0, "LAMBDA_BEND": 0},
    {"LAMBDA_RMSE": 1, "LAMBDA_POSITIONAL": 0, "LAMBDA_CHAMFER": 20, "LAMBDA_EDGE": 5, "LAMBDA_AREA": 0, "LAMBDA_BEND": 0},
    {"LAMBDA_RMSE": 1, "LAMBDA_POSITIONAL": 0, "LAMBDA_CHAMFER": 25, "LAMBDA_EDGE": 5, "LAMBDA_AREA": 0, "LAMBDA_BEND": 0},
    # {"LAMBDA_RMSE": 1, "LAMBDA_POSITIONAL": 0, "LAMBDA_CHAMFER": 10, "LAMBDA_EDGE": 5, "LAMBDA_AREA": 0, "LAMBDA_BEND": 0},
]

CONFIG_FILE = "config.py"

def update_config(params_dict):
    """
    Dynamically updates config.py with any key-value pairs provided.
    """
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    for key, value in params_dict.items():
        # Regex explanation:
        # ({key}\s*=\s*) captures the variable name and the equals sign (e.g., "BATCH_SIZE = ")
        # [^\n]+ matches everything after the equals sign until the end of the line
        # \g<1>{value} replaces the line with the captured prefix + the new value
        
        # Format string check (add quotes if the value is a string)
        val_str = f"'{value}'" if isinstance(value, str) else str(value)
        
        content = re.sub(rf'({key}\s*=\s*)[^\n]+', rf'\g<1>{val_str}', content)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)

# 2. Generate all combinations (Grid Search)
# This creates a Cartesian product: 4 model configs * 5 loss configs = 20 total runs
all_combinations = [ {**m, **l} for m, l in itertools.product(model_params, loss_params) ]

print(f"Total runs scheduled: {len(all_combinations)}\n")

# 3. Execute Sweep
for i, combo in enumerate(all_combinations, start=1):
    print(f"======================================================================")
    print(f" STARTING RUN {i}/{len(all_combinations)}")
    print(f" PARAMETERS: {combo}")
    print(f"======================================================================")
    
    # Update config.py
    update_config(combo)
    
    # Create a clean log filename using the run number and key stats
    lr = combo['LEARNING_RATE']
    bs = combo['BATCH_SIZE']
    rmse = combo['LAMBDA_RMSE']
    edge = combo['LAMBDA_EDGE']
    
    # Create a logs directory if it doesn't exist
    os.makedirs("LOGS", exist_ok=True)
    
    log_filename = f"LOGS/run{i:02d}_LR{lr}_BS{bs}_R{rmse}_E{edge}.log"
    
    # Execute training
    command = f"python train.py > {log_filename} 2>&1"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Run {i} complete! Log saved to: {log_filename}\n")
    except subprocess.CalledProcessError:
        print(f"Run {i} FAILED. Check {log_filename} for details.\n")
        # Optional: break the loop if a run fails so you don't waste hours
        # break 

print("\nAll hyperparameter sweeps completed!")