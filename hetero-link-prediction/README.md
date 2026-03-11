# Heterogeneous Link Prediction Baseline

A professional, modular baseline for Link Prediction on Heterogeneous Graphs using PyTorch and PyTorch Geometric. 
This implementation provides a simple Graph Neural Network (SAGEConv + `to_hetero` projection) to predict edges in heterogeneous networks (e.g., the MovieLens dataset).

It is designed to be easily extensible for future goals such as:
- **Metric Learning** (e.g., custom objective functions, contrastive learning)
- **Negative Sampling** (e.g., hard-negative mining strategies)
- **Graph Sparsification** (e.g., edge dropping during training)

## Project Structure

```
hetero-link-prediction/
│
├── README.md               # Project documentation
├── requirements.txt        # Pip dependencies
├── environment.yml         # Conda environment file
├── setup.py                # Package installation script
│
├── configs/
│   └── config.yaml         # Centralized configuration (dataset, model, training)
│
├── data/
│   └── dataset.py          # Data loading and graph building (using PyG)
│
├── models/
│   └── gnn_model.py        # GNN and edge decoding architecture
│
├── training/
│   └── train.py            # Train and test loop definitions
│
├── evaluation/
│   └── metrics.py          # Evaluation metrics (AUC, AP, Hits@K)
│
├── utils/
│   └── helpers.py          # Seeding, config loading, and other utilities
│
├── experiments/
│   └── run_experiment.py   # Main entry point to run the baseline
│
└── notebooks/
    └── exploration.ipynb   # Jupyter Notebook for data exploration
```

## Environment Setup (Miniconda in Windows CMD)

Follow these step-by-step instructions to set up the environment on Windows CMD:

1. **Open Windows CMD** and navigate to this project folder.

2. **Create the Conda Environment**
   ```cmd
   conda create -n hetero_link_pred python=3.9 -y
   ```

3. **Activate the Environment**
   ```cmd
   conda activate hetero_link_pred
   ```

4. **Install PyTorch** (Example using CUDA 11.8. If your system requires a different CUDA version or if you want CPU only, modify the command accordingly):
   ```cmd
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
   ```
   *(For CPU only: `conda install pytorch torchvision torchaudio cpuonly -c pytorch -y`)*

5. **Install PyTorch Geometric (PyG)**
   ```cmd
   conda install pyg -c pyg -y
   ```

6. **Install other dependencies** via pip
   ```cmd
   pip install -r requirements.txt
   ```

7. **Verify Installation**
   Check that everything is working properly:
   ```cmd
   python experiments/run_experiment.py --test-import
   ```

## Usage Instructions

To run the full pipeline (data loading, splitting, training, and evaluation), execute the following command from the root directory:

```cmd
python experiments/run_experiment.py --config configs/config.yaml
```

The script will automatically download the MovieLens dataset, build the heterogeneous graph, train the simple GNN model for user-to-movie rating prediction, and print out metrics (AUC, Average Precision, Hits@10) dynamically per epoch.

Each run now also saves research-ready artifacts under `experiments/results/<run_name>_<timestamp>/`:

- `training_history.csv` with epoch-by-epoch loss and metrics
- `test_metrics.json` with final held-out evaluation results
- `training_curves.png` with publication-style loss and AUC curves
