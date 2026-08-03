from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss
)

def evaluate_model(y_test, predictions, probabilities):

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    specificity = tn / (tn + fp)

    results = {

        "Accuracy":
            accuracy_score(y_test, predictions),

        "ROC AUC":
            roc_auc_score(y_test, probabilities),

        "PR AUC":
            average_precision_score(y_test, probabilities),

        "F1":
            f1_score(y_test, predictions),

        "MCC":
            matthews_corrcoef(y_test, predictions),

        "Sensitivity":
            recall_score(y_test, predictions),

        "Specificity":
            specificity,

        "Balanced Accuracy":
            balanced_accuracy_score(
                y_test,
                predictions
            ),

        "Brier Score":
            brier_score_loss(
                y_test,
                probabilities
            ),
        "Confusion Matrix":
            confusion_matrix(
                y_test,
                predictions
            ),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

    return results


def evaluate_outer_fold(estimator, X_test, y_test, threshold):
  test_probabilities = estimator.predict_proba(
        X_test
    )[:, 1]

  test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

  ### Evaluate fold ###
  fold_metrics = evaluate_model(

        y_test= y_test,
        predictions= test_predictions,
        probabilities= test_probabilities

    )
  return fold_metrics, test_predictions, test_probabilities