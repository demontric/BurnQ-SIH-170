import numpy as np
import xgboost as xgb
from sklearn.metrics import make_scorer

# =============================================================================
# Cost-Sensitive Learning for Burn-In Anomaly Detection & Drift Prediction
# =============================================================================
# Flaw addressed: Standard ML models optimize for overall accuracy or standard MSE,
# treating False Positives and False Negatives equally. In space-grade manufacturing, 
# missing a defective part (False Negative) is catastrophic, while rejecting a good
# part (False Positive) is an acceptable cost.
#
# Fix: Modify the loss function to heavily penalize under-predictions (False Negatives).
# =============================================================================

def asymmetric_mse_objective(preds, dtrain):
    """
    Cost-sensitive custom objective function for XGBoost (Drift Predictor).
    Penalizes under-predictions (where actual drift is higher than predicted) 
    100x more than over-predictions to ensure no catastrophic failures escape.
    """
    labels = dtrain.get_label()
    # residual = predicted - actual
    # If predicted < actual (under-prediction / false negative), residual is negative.
    res = preds - labels
    
    # Asymmetric penalty: 100x penalty if under-predicting
    penalty = np.where(res < 0, 100.0, 1.0)
    
    # Gradient (first derivative)
    grad = penalty * res
    
    # Hessian (second derivative)
    hess = penalty
    
    return grad, hess

def asymmetric_mse_metric(preds, dtrain):
    """
    Custom evaluation metric for tracking cost-sensitive MSE during training.
    """
    labels = dtrain.get_label()
    res = preds - labels
    penalty = np.where(res < 0, 100.0, 1.0)
    cost = np.mean(penalty * (res ** 2))
    return 'asym_mse', cost

# Example Usage:
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
# params = {
#     'max_depth': 4,
#     'learning_rate': 0.1
# }
# model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=100,
#     obj=asymmetric_mse_objective,
#     feval=asymmetric_mse_metric,
#     evals=[(dtrain, 'train'), (dtest, 'test')]
# )

def cost_sensitive_isolation_forest_threshold(anomaly_scores, risk_tolerance=50.0):
    """
    Cost-sensitive thresholding for Isolation Forest (Anomaly Detection).
    Instead of a standard probability > 0.5 threshold, we dynamically adjust 
    the decision boundary based on a risk tolerance slider from the UI.
    
    risk_tolerance: 1 to 100. Lower value means less tolerance for risk 
                    (more sensitive, catches more False Negatives).
    """
    # Base threshold
    base_threshold = -0.1
    
    # Shift threshold down to classify more points as anomalies
    modifier = (50.0 - risk_tolerance) * 0.005 
    
    return base_threshold - modifier
