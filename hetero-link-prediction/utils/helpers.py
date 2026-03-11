import random
import numpy as np
import torch
import yaml

def set_seed(seed=42):
    """
    Sets global seeds reliably to establish deterministic modeling workflows 
    and reliable experimentation output constraints.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Parses root parameters securely off standardized YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
