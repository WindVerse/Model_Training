import os
import torch

# Dataset Properties
DATASET_VERSION = 6
TARGET_TYPE = "accelerations"                    # displacements, velocity_differences, accelerations
EXIST_TOPOLOGY = True
TRAIN_RATIO = 0.8
ITERATION_COUNT = 100
FPS = 10
MAX_FRAMES = 300
HEIGHT = 20
WIDTH = 30
NODE_DIM = 6                                     # 3-[Pos_x, Pos_y, Pos_z], 6-[Pos_x, Pos_y, Pos_z, Vel_x, Vel_y, Vel_z]
WIND_DIM = 3
EDGE_DIM = 7                                     # [Rel_Pos(3), Rel_Vel(3), Dist(1)]
NUM_VERTICES = HEIGHT*WIDTH
BASE_DATASET_PATH = "../../datasets/"

# Hyperparameters
MODEL = "GNN"                                    # 'GNN', add more later
LOSS = "physicsLoss"                             # 'physicsLoss', add more later
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 4

SEQUENCE_LENGTH = 10
NO_MLP_HIDDEN_LAYERS = 3
NO_GNN_LAYERS = 3
HIDDEN_DIM = 64
GNN_AGGREGATION = "add"                          # 'add', 'mean', 'max'
DROPOUT_RATE = 0.1
ACTIVATION = 'SiLU'                              # 'ReLU', 'SiLU', 'Tanh', 'LeakyReLU'
USE_LAYER_NORM = True

# Loss Hyperparameters
LAMBDA_WARP = 1.0      # Weight for Spring Constraint (Edge Length)
LAMBDA_SMOOTH = 0.1    # Weight for Smoothness
LAMBDA_PIN = 50.0      # Weight for Pinned Nodes (Pole)

# Optimizer
OPTIMIZER = 'Adam'          # Options: 'Adam', 'SGD', 'RMSprop'
LEARNING_RATE = 1e-4        # Initial Learning Rate
WEIGHT_DECAY = 1e-5         # L2 Regularization (Prevents exploding weights)
MOMENTUM = 0.9              # Used only for SGD

# Scheduler
SCHEDULER = 'ReduceLROnPlateau' # Options: 'ReduceLROnPlateau', 'StepLR', 'None'
# Scheduler Specifics
# 1. ReduceLROnPlateau (Reduces LR when validation loss stops improving)
SCHEDULER_FACTOR = 0.5      # Multiply LR by this factor
SCHEDULER_PATIENCE = 5      # How many epochs to wait before reducing
# 2. StepLR (Reduces LR every X epochs)
SCHEDULER_STEP_SIZE = 10    # Decay every 10 epochs
SCHEDULER_GAMMA = 0.1       # Decay rate




# Auto

DATASET_DIR = os.path.join(BASE_DATASET_PATH, str(DATASET_VERSION))
FLAG_DIR = os.path.join(DATASET_DIR, "flags")
WIND_DIR = os.path.join(DATASET_DIR, "winds")
TARGET_DIR = os.path.join(DATASET_DIR, "targets", TARGET_TYPE)
TOPOLOGY_PATH = os.path.join(DATASET_DIR, "topology", "topology_edge_index.npy")

DELTA_T = 1.0 / FPS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Config loaded. Device: {DEVICE}")