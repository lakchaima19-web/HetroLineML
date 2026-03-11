import sys
import os
import argparse

# Enable absolute package import resolutions organically across environments.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.dataset import get_dataset
from models.gnn_model import HeteroLinkPredictionModel
from training.train import evaluate_epoch, train_epoch
from utils.helpers import set_seed, load_config
from utils.visualization import (
    build_run_dir,
    ensure_dir,
    plot_training_curves,
    save_history_csv,
    save_metrics_json,
)

def main():
    parser = argparse.ArgumentParser(description="Baseline Runner for Heterogeneous Link Prediction")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Local YAML path')
    parser.add_argument('--test-import', action='store_true', help='Test underlying application library imports and verify system')
    args = parser.parse_args()

    if args.test_import:
        print("Imports successful! Core libraries dynamically bound. Environment logic sets properly.")
        return

    config = load_config(args.config)
    experiment_config = config.get('experiment', {})
    
    # Establishing Baseline Reproducibility Contexts
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing over core framework backend device: {device}")

    # 1. Pipeline Initializer / Data Loading
    print(f"Loading native graph topology definition map: {config['dataset']['name']}")
    train_data, val_data, test_data, _ = get_dataset(
        data_dir=config['dataset']['data_dir'], 
        dataset_name=config['dataset']['name']
    )
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Establish edge prediction target specifically linking Users and Movies implicitly via interactions
    edge_type = ("user", "rates", "movie")

    # 2. Lazy Initialization for Message Passing Model State Dimensions Binding 
    print("Initialising abstract target decoder boundaries...")
    model = HeteroLinkPredictionModel(
        hidden_channels=config['model']['hidden_channels'], 
        metadata=train_data.metadata()
    ).to(device)

    # Pass dummy batch forward to explicitly build PyG specific lazily instanced layer components sizes properly
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict, 
              train_data[edge_type].edge_label_index, edge_type)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )

    epochs = config['training']['epochs']
    k = config['evaluation']['hits_k']
    output_root = experiment_config.get('output_dir', os.path.join('experiments', 'results'))
    run_dir = build_run_dir(output_root, experiment_config.get('run_name'))
    ensure_dir(run_dir)
    history = []
    
    print("\nStarting network backpropagation cycles...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, train_data, edge_type)
        train_metrics = evaluate_epoch(model, train_data, edge_type, k=k)
        val_metrics = evaluate_epoch(model, val_data, edge_type, k=k)
        history.append({
            'epoch': epoch,
            'train_loss': loss,
            'train_auc': train_metrics['AUC'],
            'train_ap': train_metrics['AP'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['AUC'],
            'val_ap': val_metrics['AP'],
            f'val_hits_at_{k}': val_metrics[f'Hits@{k}'],
        })
        
        print(f"[Epoch {epoch:03d}/{epochs:03d}] Loss: {loss:.4f} "
              f"| Train AUC: {train_metrics['AUC']:.4f} "
              f"| Val AUC: {val_metrics['AUC']:.4f} "
              f"| Val Loss: {val_metrics['loss']:.4f} "
              f"| Val AP: {val_metrics['AP']:.4f} "
              f"| Val Hits@{k}: {val_metrics[f'Hits@{k}']:.4f}")

    print("\n--- Final Framework Edge Target Testing Results ---")
    test_metrics = evaluate_epoch(model, test_data, edge_type, k=k)
    print(f"Test AUC      : {test_metrics['AUC']:.4f}")
    print(f"Test AP       : {test_metrics['AP']:.4f}")
    print(f"Test Hits@{k}  : {test_metrics[f'Hits@{k}']:.4f}")
    print(f"Test Loss     : {test_metrics['loss']:.4f}")

    save_history_csv(history, os.path.join(run_dir, 'training_history.csv'))
    save_metrics_json(test_metrics, os.path.join(run_dir, 'test_metrics.json'))
    plot_training_curves(
        history=history,
        output_path=os.path.join(run_dir, 'training_curves.png'),
        dataset_name=config['dataset']['name'],
        model_name=experiment_config.get('model_name', 'HeteroLinkPredictionModel'),
    )
    print(f"\nArtifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()
