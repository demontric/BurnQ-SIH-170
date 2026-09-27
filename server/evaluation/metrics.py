import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

def classification_metrics(y_true, y_pred):
    """Calculates strict cost-sensitive classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Recall": float(recall),
        "Precision": float(precision),
        "FNR": float(fnr),
        "FPR": float(fpr)
    }

def regression_metrics(y_true, y_pred):
    """Calculates standard regression fit metrics."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t, y_p = y_true[valid], y_pred[valid]
    
    if len(y_t) == 0:
        return {"MAE": None, "RMSE": None, "R2": None}
        
    return {
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "R2": float(r2_score(y_t, y_p))
    }