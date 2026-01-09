# WindVerse ML Training

This repository contains the machine learning training pipeline for **WindVerse**. This module handles dataset management, model configuration, and training execution for simulating flag dynamics using Graph Neural Networks (GNNs).

## 📋 Prerequisites

* **OS:** Linux or Windows
* **Python:** 3.11.14
* **Conda:** For environment management
* **Hardware:** NVIDIA GPU recommended (CUDA supported)

---

## 🚀 Setup & Installation

Follow these steps to set up the environment and start training from scratch.

### 1. Project Initialization
Create a root directory and clone the repository inside it.

```bash
# 1. Create a root folder (e.g., WindVerse)
mkdir WindVerse
cd WindVerse

# 2. Clone the repository inside root
git clone https://github.com/WindVerse/Model_Training.git
```

download the dataset from here [ will give a link later ]
extract the dataset into root filder.
```bash
WindVerse_Root/
├── datasets/
│   └── 1/              <-- Your Renamed Dataset Folder
│       ├── flags/      (.npy files)
│       ├── winds/      (.npy files)
│       └── topology/   (edges.npy)
└── Model_Training/     <-- Cloned Repository
```
### 1. Project Initialization

now open a terminal in "Model_Training/main/" folder.
```bash
conda create -n windverse_ml python=3.11.14 -y

conda activate windverse_ml

# IF WINDOWS
pip install -r requirements.txt

# IF LINUX
pip install -r requirementsML.txt

python train.py
```
