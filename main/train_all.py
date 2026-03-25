import os
import re
import subprocess

# 1. Define your combinations here
# Add or remove combinations as needed
hyperparameter_grid = [
    {"RMSE": 1, "CHAMFER": 50, "EDGE": 50, "AREA": 0, "BEND": 0},
    {"RMSE": 1, "CHAMFER": 10, "EDGE": 10, "AREA": 0, "BEND": 0},
    {"RMSE": 1, "CHAMFER": 50, "EDGE": 10, "AREA": 0, "BEND": 0},
    {"RMSE": 1, "CHAMFER": 10, "EDGE": 50, "AREA": 0, "BEND": 0},
    {"RMSE": 1, "CHAMFER": 50, "EDGE": 50, "AREA": 2, "BEND": 4},
    {"RMSE": 1, "CHAMFER": 10, "EDGE": 10, "AREA": 1, "BEND": 2},
    {"RMSE": 1, "CHAMFER": 50, "EDGE": 10, "AREA": 1, "BEND": 2},
    {"RMSE": 1, "CHAMFER": 10, "EDGE": 50, "AREA": 1, "BEND": 2},
]

CONFIG_FILE = "config.py"

def update_config(rmse, chamfer, edge, area, bend):
    """Reads config.py, updates the lambdas using regex, and overwrites the file."""
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Regex search and replace for the specific variables
    content = re.sub(r'LAMBDA_RMSE\s*=\s*[\d.]+', f'LAMBDA_RMSE = {rmse}', content)
    content = re.sub(r'LAMBDA_CHAMFER\s*=\s*[\d.]+', f'LAMBDA_CHAMFER = {chamfer}', content)
    content = re.sub(r'LAMBDA_EDGE\s*=\s*[\d.]+', f'LAMBDA_EDGE = {edge}', content)
    content = re.sub(r'LAMBDA_AREA\s*=\s*[\d.]+', f'LAMBDA_AREA = {area}', content)
    content = re.sub(r'LAMBDA_BEND\s*=\s*[\d.]+', f'LAMBDA_BEND = {bend}', content)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)

# 2. Loop through the combinations
for combo in hyperparameter_grid:
    r = combo["RMSE"]
    c = combo["CHAMFER"]
    e = combo["EDGE"]
    a = combo["AREA"]
    b = combo["BEND"]
    
    print(f"\n======================================================================")
    print(f"   STARTING RUN: RMSE={r} | CHAMFER={c} | EDGE={e} | AREA={a} | BEND={b}")
    print(f"======================================================================")
    
    # Update the config file
    update_config(r, c, e, a, b)
    
    # Create a unique log file for this specific run
    log_filename = f"training_R{r}_C{c}_E{e}_A{a}_B{b}.log"
    
    # Run train.py sequentially (this blocks until train.py finishes)
    # Using shell=True allows us to use the standard '>' redirection
    command = f"python train.py > {log_filename} 2>&1"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"? Run complete! Log saved to: {log_filename}")
    except subprocess.CalledProcessError:
        print(f"? Run FAILED. Check {log_filename} for details.")
        # Decide if you want to break the loop on failure or continue
        # break 

print("\n?? All hyperparameter sweeps completed!")