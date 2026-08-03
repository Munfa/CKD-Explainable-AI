import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from utils.models import get_base_models, get_models

scaled_models = {
    "Logistic Regression",
    "SVM",
    "MLP"
}

# ==========================================================
# Scale selected models and create pipeline for all
# ==========================================================
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


# ==========================================================
# Find best threshold after training
# ==========================================================
# def find_best_threshold(y_true, y_prob, thresholds=np.arange(0.01, 1.00, 0.01)):
#     best_threshold = 0.5
#     best_mcc = -1

#     for threshold in thresholds:
#         y_pred = (y_prob >= threshold).astype(int)
#         mcc = matthews_corrcoef(y_true, y_pred)

#         if mcc > best_mcc:
#             best_mcc = mcc
#             best_threshold = threshold
        
#     return best_threshold, best_mcc

def select_best_threshold(model_name, best_param, X_train, y_train, inner_cv, groups=None):
  ### Generate inner out-of-fold probabilities ###
  inner_probabilities = np.zeros(len(y_train))

  # Generate inner folds
  if groups is None:
      splits = inner_cv.split(X_train, y_train)
  else:
      splits = inner_cv.split(X_train, y_train, groups)

  for inner_train_idx, inner_valid_idx in splits:

        # Split inner fold
        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train.iloc[inner_train_idx]

        X_inner_valid = X_train.iloc[inner_valid_idx]

        # Create a fresh pipeline
        if model_name in ["Majority Dummy", "Stratified Dummy"]:
          model = get_base_models()[model_name]
        else:
          model = get_models()[model_name]
        pipeline = create_pipeline(model_name, model)

        # Apply tuned parameters
        pipeline.set_params(**best_param)

        # Train
        pipeline.fit(X_inner_train, y_inner_train)

        # Predict probabilities
        inner_probabilities[inner_valid_idx] = (
            pipeline.predict_proba(X_inner_valid)[:, 1]
        )

  ##### Find best threshold #####
  thresholds=np.arange(0.01, 1.00, 0.01)
  best_threshold = 0.5
  best_mcc = -1
  for threshold in thresholds:
          y_pred = (inner_probabilities >= threshold).astype(int)
          mcc = matthews_corrcoef(y_train, y_pred)
  
          if mcc > best_mcc:
              best_mcc = mcc
              best_threshold = threshold
          
  return best_threshold, best_mcc
  

# ==========================================================
# Save predictions per fold 
# ==========================================================
def save_fold_predictions(save_dir, model_name, fold_number, estimator, best_params, threshold, 
                          test_indices, y_true, y_prob, y_pred):
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
        model_dir,
        f"fold{fold_number}.joblib"
    )

    joblib.dump(output, filename)
    return filename