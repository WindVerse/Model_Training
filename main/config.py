import os
import torch

# Test or Not
IS_TEST = True



#########################################
########### Dataset Properties ##########
#########################################

if IS_TEST:
    DATASET_VERSION = 0
else:
    DATASET_VERSION = 6
TARGET_TYPE = "accelerations"                    # displacements, accelerations
EXIST_TOPOLOGY = True
TRAIN_RATIO = 0.8
if IS_TEST:
    ITERATION_COUNT = 5
else:
    ITERATION_COUNT = 100
FPS = 10
if IS_TEST:
    MAX_FRAMES = 30
else:
    MAX_FRAMES = 300
HEIGHT = 20
WIDTH = 30
NODE_DIM = 6                                     # 3-[Pos_x, Pos_y, Pos_z], 6-[Pos_x, Pos_y, Pos_z, Vel_x, Vel_y, Vel_z]
WIND_DIM = 3
EDGE_DIM = 7                                     # [Rel_Pos(3), Rel_Vel(3), Dist(1)]
NUM_VERTICES = HEIGHT*WIDTH
BASE_DATASET_PATH = "../../datasets/"





#########################################
########### Model Hyperparameters #######
#########################################

MODEL = "GNN"                                    # 'GNN', 'SNN', 'LSTM_CNN'
LOSS = "physicsLoss"                                  # 'physicsLoss', add more later

ADD_NOISE = False
NOISE_STD = 0.001

FLAG_ENABLED = False                   # Free Large-scale Adversarial Augmentation on Graphs (flag) for worst case noise addition
FLAG_STEPS = 3        # M             # number of forward passes to get the worst case loss
FLAG_STEP_SIZE = 1e-3 # α             # step size for a forward pass

if IS_TEST:
    EPOCHS = 2
else:
    EPOCHS = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 4
WARMUP_EPOCHS = 0                      # Should be less than total epochs.

SEQUENCE_LENGTH = 1                             # make 1 for frame-by-frame training, >1 for sequence training (e.g., LSTM)


if IS_TEST:
    NO_MLP_HIDDEN_LAYERS = 5
    NO_GNN_LAYERS = 5
    NO_LSTM_LAYERS = 2
    CNN_CHANNELS = [16, 32, 64] # Try [16, 32] for shallower, or [16, 32, 64, 128] for deeper (Only for LSTM_CNN)
    HIDDEN_DIM = 128
else:
    NO_MLP_HIDDEN_LAYERS = 5
    NO_GNN_LAYERS = 5
    NO_LSTM_LAYERS = 3
    CNN_CHANNELS = [16, 32, 64] # Try [16, 32] for shallower, or [16, 32, 64, 128] for deeper (Only for LSTM_CNN)
    HIDDEN_DIM = 128
GNN_AGGREGATION = "add"                          # 'add', 'mean', 'max'
DROPOUT_RATE = 0.1
ACTIVATION = 'ReLU'                              # 'ReLU', 'SiLU', 'Tanh', 'LeakyReLU'
USE_LAYER_NORM = True





##########################################
########### Loss Hyperparameters #########
##########################################

LAMBDA_RMSE = 1
LAMBDA_POSITIONAL = 1
LAMBDA_CHAMFER = 0
LAMBDA_EDGE = 10
LAMBDA_SMOOTH = 0.0    # Weight for Smoothness
LAMBDA_AREA = 0
LAMBDA_BEND = 0
LAMBDA_PIN = 0.0       # Weight for Pinned Nodes (Pole)





##########################################
########### Optimizer Settings ###########
##########################################

OPTIMIZER = 'Adam'          # Options: 'Adam', 'SGD', 'RMSprop'
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5         # L2 Regularization (Prevents exploding weights)
MOMENTUM = 0.9              # Used only for SGD




##########################################
########### Scheduler Settings ###########
##########################################

SCHEDULER = 'ReduceLROnPlateau' # Options: 'ReduceLROnPlateau', 'StepLR', 'None'
# Scheduler Specifics
# 1. ReduceLROnPlateau (Reduces LR when validation loss stops improving)
SCHEDULER_FACTOR = 0.5      # Multiply LR by this factor
SCHEDULER_PATIENCE = 5      # How many epochs to wait before reducing
SCHEDULER_MODE = 'min'      # 'min' or 'max' based on monitored metric
# 2. StepLR (Reduces LR every X epochs)
SCHEDULER_STEP_SIZE = 10    # Decay every 10 epochs
SCHEDULER_GAMMA = 0.1       # Decay rate




# Auto

DATASET_DIR = os.path.join(BASE_DATASET_PATH, str(DATASET_VERSION))
FLAG_DIR = os.path.join(DATASET_DIR, "flags")
WIND_DIR = os.path.join(DATASET_DIR, "winds")
TARGET_DIR = os.path.join(DATASET_DIR, "targets", TARGET_TYPE)
TOPOLOGY_PATH = os.path.join(DATASET_DIR, "topology", "topology_edge_index.npy")
FACES_PATH = os.path.join(DATASET_DIR, "topology", "topology_faces.npy")


DELTA_T = 1.0 / FPS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Config loaded. Device: {DEVICE}")