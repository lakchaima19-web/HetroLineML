import torch
import torch.nn.functional as F
from evaluation.metrics import evaluate_link_prediction

def train_epoch(model, optimizer, data, edge_type):
    """
    Trains the model for one complete epoch using standard full-batch processing.
    """
    model.train()
    optimizer.zero_grad()
    
    # Predict output logits for supervised training edges
    out = model(data.x_dict, data.edge_index_dict, 
                data[edge_type].edge_label_index, edge_type)
    
    target = data[edge_type].edge_label.float()
    
    # Binary Cross Entropy with Logits Loss explicitly handles prediction properly 
    loss = F.binary_cross_entropy_with_logits(out, target)
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate_epoch(model, data, edge_type, k=10):
    """
    Evaluates model performance given a configured subgraph split.
    Uses specified metrics logic imported from standard custom implementation.
    """
    model.eval()
    
    out = model(data.x_dict, data.edge_index_dict, 
                data[edge_type].edge_label_index, edge_type)
    
    # Apply sigmoid to extract actual binary prediction probabilities
    pred_probs = torch.sigmoid(out)
    target = data[edge_type].edge_label.float()
    loss = F.binary_cross_entropy_with_logits(out, target)
    
    metrics = evaluate_link_prediction(target, pred_probs, k=k)
    metrics["loss"] = loss.item()
    return metrics

@torch.no_grad()
def test(model, data, edge_type, k=10):
    """
    Backward-compatible alias for evaluation over a full split.
    """
    return evaluate_epoch(model, data, edge_type, k=k)
