import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

scaled_models = {
    "Logistic Regression",
    "SVM",
    "MLP"
}

def create_pipeline(model_name, estimator):
    if model_name in scaled_models:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator)
        ])
    else:
        pipeline = Pipeline([
            ("model", estimator)
        ])
    return pipeline

def find_best_threshold(y_true, y_prob, thresholds=np.arange(0.01, 1.00, 0.01)):
    best_threshold = 0.5
    best_mcc = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
        
    return best_threshold, best_mcc

def save_fold_predictions(save_dir, model_name, fold_number, estimator, best_params, threshold, test_indices,
                          y_true, y_prob, y_pred):
    model_dir = os.path.join(
        save_dir, 
        model_name.replace(" ","_")
        )
    
    os.makedirs(
        model_dir,
        exist_ok=True
    )

    output = {

        "fold": fold_number,

        "best_estimator": estimator,

        "best_params": best_params,

        "threshold": threshold,

        "test_indices": test_indices,

        "y_true": y_true,

        "y_prob": y_prob,

        "y_pred": y_pred

    }

    filename = os.path.join(
        model_name,
        f"fold{fold_number}.joblib"
    )

    joblib.dump(output, filename)