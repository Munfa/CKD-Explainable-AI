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

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:,1]

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
            )
    }

    return results