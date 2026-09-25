import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest

def asymmetric_mse_objective(preds, dtrain):
    """
    Cost-sensitive custom objective function for XGBoost (Module B).
    Penalizes under-predictions (False Negatives) 100x more than over-predictions
    to prevent catastrophic component failures from escaping.
    """
    labels = dtrain.get_label()
    res = preds - labels
    
    # 100x penalty if predicted < actual (under-prediction / False Negative)
    penalty = np.where(res < 0, 100.0, 1.0)
    
    # Apply penalty to gradient to force massive correction steps in the trees
    grad = penalty * res
    
    # Leave Hessian unscaled by the penalty. 
    # If both are scaled, the penalty cancels out in XGBoost's leaf weight formula.
    hess = np.ones_like(res)
    
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

def train_drift_predictor(X_train, y_train, X_test, y_test, params=None, num_boost_round=100):
    """
    Trains the XGBoost drift predictor using the asymmetric objective.
    X_train should contain rolling-baseline adjusted 0h and 24h features.
    """
    if params is None:
        params = {
            'max_depth': 4,
            'learning_rate': 0.1,
            'disable_default_eval_metric': 1  # Required when using a custom metric
        }
        
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=asymmetric_mse_objective,
        custom_metric=asymmetric_mse_metric,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    return model

def cost_sensitive_isolation_forest_threshold(anomaly_scores, risk_tolerance=50.0):
    """
    Dynamic thresholding for Isolation Forest (Module A).
    
    risk_tolerance: 1 to 100. 
    1 = Catch everything (High False Positives, Zero False Negatives).
    100 = Catch only massive outliers (Low False Positives, High False Negatives).
    """
    # IsolationForest decision_function: negative scores are anomalies, positive are normal.
    base_threshold = -0.1
    
    # Low risk tolerance increases the threshold (closer to 0 or positive)
    modifier = (50.0 - risk_tolerance) * 0.005 
    
    # Add the modifier to shift the decision boundary
    dynamic_threshold = base_threshold + modifier
    
    # Flags as True (Anomaly) if the score is lower than the dynamic threshold
    flags = anomaly_scores < dynamic_threshold
    
    return dynamic_threshold, flags

def detect_anomalies(features, model=None, risk_tolerance=50.0):
    """
    Evaluates current lot features against a trained Isolation Forest.
    Features must be normalized against the dynamic rolling lot average before this step.
    """
    if model is None:
        # Fallback initialization; in production, load the pre-trained/warmed model
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(features)
        
    anomaly_scores = model.decision_function(features)
    threshold, flags = cost_sensitive_isolation_forest_threshold(anomaly_scores, risk_tolerance)
    
    return flags, anomaly_scores, threshold