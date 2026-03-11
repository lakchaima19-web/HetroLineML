import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens

def get_dataset(data_dir='./data_files', dataset_name='MovieLens'):
    """
    Downloads and loads the heterogeneous dataset.
    Creates a random link split for training, validation, and testing.
    """
    if dataset_name == 'MovieLens':
        # Load the MovieLens dataset (a typical bipartite user-movie graph)
        dataset = MovieLens(root=os.path.join(data_dir, 'MovieLens'))
        data = dataset[0]
        
        # Depending on PyG version, user and movie features might be missing.
        # We manually initialize basic random node features if they don't exist.
        if 'x' not in data['user']:
            num_users = data['user'].num_nodes
            # Simple standard normal features representing latent aspects
            data['user'].x = torch.randn((num_users, 64))
            
        if 'x' not in data['movie']:
            num_movies = data['movie'].num_nodes
            data['movie'].x = torch.randn((num_movies, 64))

        # We want to predict ("user", "rates", "movie") edges.
        # Adding reverse edges explicitly via PyG built-in transform for robust message passing
        data = T.ToUndirected()(data)
        
        # Ensure we are doing **link prediction**, not rating prediction by removing multiclass labels
        # RandomLinkSplit will automatically set positive links to 1 and negative links to 0.
        if 'edge_label' in data["user", "rates", "movie"]:
            del data["user", "rates", "movie"]['edge_label']
        if 'edge_label' in data["movie", "rev_rates", "user"]:
            del data["movie", "rev_rates", "user"]['edge_label']


        # Splitting edges for link prediction
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,     # For message passing edges vs supervision edges
            neg_sampling_ratio=1.0,       # 1 negative training sample per positive
            add_negative_train_samples=True, 
            edge_types=[("user", "rates", "movie")],
            rev_edge_types=[("movie", "rev_rates", "user")], 
        )
        train_data, val_data, test_data = transform(data)
        
        return train_data, val_data, test_data, dataset
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
