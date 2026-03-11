from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import numpy as np

def calculate_hits_at_k(y_true, y_pred_probs, k=10):
    """
    Computes a simplified validation Hits@K strategy mapping.
    This counts how many of the true edges were actually localized into the 
    top k predictions from the current evaluation split.
    """
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Edge case: no positive samples or prediction bounds
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0
        
    # Get indices of the upper k highest probability edges predicted
    top_k_indices = np.argsort(y_pred_probs)[-k:]
    
    # Intersect with absolute true labels
    hits = np.sum(y_true[top_k_indices])
    
    # Standard normalization bounding
    return hits / min(k, total_positives)

def evaluate_link_prediction(y_true, y_pred_probs, k=10):
    """
    Wraps calculations for generic link prediction metrics logic:
    - ROC AUC
    - Average Precision
    - Hits@K
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_probs_np = y_pred_probs.cpu().numpy()

    # Fallback default returns for undefined spaces when batch distributions lack negatives or positives
    if len(np.unique(y_true_np)) == 1:
        return {"AUC": 0.0, "AP": 0.0, f"Hits@{k}": 0.0}

    auc = roc_auc_score(y_true_np, y_pred_probs_np)
    ap = average_precision_score(y_true_np, y_pred_probs_np)
    hits_k = calculate_hits_at_k(y_true_np, y_pred_probs_np, k)

    return {"AUC": auc, "AP": ap, f"Hits@{k}": hits_k}
