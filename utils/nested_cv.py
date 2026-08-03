
import os
import joblib
from utils.optimization import tune_model
from utils.pipeline import create_pipeline, select_best_threshold, save_fold_predictions
from utils.evaluation import evaluate_outer_fold

def run_outer_fold_loop(exp_dict, model_name, model, X, y, outer_cv, inner_cv,
                        FOLD_DIR, MODEL_DIR, scoring, groups=None, base_name=None):
  for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y, groups),
            start=1):

        ################# Create train/test split #################
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        if groups is None:
          group_train = None
        else:
          group_train = groups.iloc[train_idx]

        ################# Build pipeline #################
        pipeline = create_pipeline(
            model_name=model_name,
            estimator=model
        )

        ################# Get best pipeline and parameters #################
        best_pipeline, best_params = tune_model(
            model_name, pipeline, X_train, y_train,
            inner_cv, scoring, group_train)

        if base_name is None:
          base_name = model_name
        exp_dict[base_name]["best_params"].append(
            best_params
        )

        ################# Find best threshold #################
        threshold, best_inner_mcc = select_best_threshold(model_name, best_params, X_train, y_train,
                                                          inner_cv, group_train)
        exp_dict[base_name]["thresholds"].append(
            threshold
        )
        exp_dict[base_name]["best_inner_mcc"].append(
            best_inner_mcc
        )

        ################# Retrain on the outer training fold #################
        best_pipeline.fit(X_train, y_train)

        ################# Evaluate fold #################
        fold_metrics, y_pred, y_prob = evaluate_outer_fold(
            best_pipeline, X_test, y_test, threshold
        )

        fold_metrics['Fold'] = fold
        fold_metrics['Threshold'] = threshold
        exp_dict[base_name]["fold_metrics"].append(
            fold_metrics
        )

        ################# Save fold path #################
        fold_path = save_fold_predictions(FOLD_DIR, model_name, fold, best_pipeline, best_params,
                                          threshold, test_idx, y_test, y_prob, y_pred)
        exp_dict[base_name]["saved_folds"].append(
            fold_path
        )

        ################# Save trained model #################
        model_dir = os.path.join(
          MODEL_DIR,
          model_name.replace(" ","_")
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )
        model_path = os.path.join(model_dir, f"{model_name}_fold{fold}.joblib")

        joblib.dump(best_pipeline, model_path)
        exp_dict[base_name]["saved_models"].append(
            model_path
        )
