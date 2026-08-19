from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42

def get_base_models():
    baseline_models = {
        "Majority Dummy": DummyClassifier(strategy="most_frequent"),

        "Stratified Dummy": DummyClassifier(strategy="stratified",
                                            random_state=RANDOM_STATE),
    }
    return baseline_models

def get_age_baseline():
    age_baseline = {
        "Logistic Regression":
            LogisticRegression(max_iter=5000,
                            random_state=RANDOM_STATE)
    }
    return age_baseline

def get_models():
    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=5000,
                            random_state=RANDOM_STATE),

        "Decision Tree":
            DecisionTreeClassifier(random_state=RANDOM_STATE),

        "Random Forest":
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
                ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ),

        "LightGBM":
            LGBMClassifier(verbose=-1, random_state=RANDOM_STATE),

        "CatBoost":
            CatBoostClassifier(
                verbose=0,
                random_state=RANDOM_STATE
            ),

        "SVM":
            SVC(probability=True, 
                kernel='rbf',
                random_state=RANDOM_STATE),

        "MLP":
            MLPClassifier(
                max_iter=5000,
                early_stopping=True,
                random_state=RANDOM_STATE
            )
    }
    return models