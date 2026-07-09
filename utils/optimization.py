from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform

# ==========================================================
# Hyperparameter Search Space
# ==========================================================
def get_param_dist():
    PARAM_DISTRIBUTIONS = {

        # ------------------------------------------------------
        # Logistic Regression
        # ------------------------------------------------------

        "Logistic Regression": {

            "model__C":

                loguniform(1e-3, 100),

            "model__penalty":

                ["l2"],

            "model__solver":

                ["lbfgs"],

            "model__class_weight":

                [None, "balanced"]

        },

        # ------------------------------------------------------
        # Decision Tree
        # ------------------------------------------------------

        "Decision Tree": {

            "model__max_depth":

                randint(2, 25),

            "model__min_samples_split":

                randint(2, 20),

            "model__min_samples_leaf":

                randint(1, 10),

            "model__class_weight":

                [None, "balanced"]

        },

        # ------------------------------------------------------
        # Random Forest
        # ------------------------------------------------------

        "Random Forest": {

            "model__n_estimators":

                randint(100, 500),

            "model__max_depth":

                randint(3, 25),

            "model__max_features":

                ["sqrt", "log2", None],

            "model__min_samples_leaf":

                randint(1, 10),

            "model__class_weight":

                [None, "balanced"]

        },

        # ------------------------------------------------------
        # XGBoost
        # ------------------------------------------------------

        "XGBoost": {

            "model__n_estimators":

                randint(100, 500),

            "model__learning_rate":

                uniform(0.01, 0.25),

            "model__max_depth":

                randint(3, 10),

            "model__min_child_weight":

                randint(1, 10),

            "model__subsample":

                uniform(0.6, 0.4),

            "model__colsample_bytree":

                uniform(0.6, 0.4),

            "model__reg_alpha":

                uniform(0, 5),

            "model__reg_lambda":

                uniform(0.5, 5),

            "model__scale_pos_weight":

                [1, 2, 3, 5]

        },

        # ------------------------------------------------------
        # LightGBM
        # ------------------------------------------------------

        "LightGBM": {

            "model__n_estimators":

                randint(100, 500),

            "model__learning_rate":

                uniform(0.01, 0.25),

            "model__num_leaves":

                randint(20, 80),

            "model__max_depth":

                randint(3, 12),

            "model__min_child_samples":

                randint(10, 40),

            "model__subsample":

                uniform(0.6, 0.4),

            "model__colsample_bytree":

                uniform(0.6, 0.4),

            "model__class_weight":

                [None, "balanced"]

        },

        # ------------------------------------------------------
        # CatBoost
        # ------------------------------------------------------

        "CatBoost": {

            "model__depth":

                randint(4, 10),

            "model__learning_rate":

                uniform(0.01, 0.25),

            "model__iterations":

                randint(100, 500),

            "model__l2_leaf_reg":

                uniform(1, 9),

            "model__auto_class_weights":

                [None, "Balanced"]

        },

        # ------------------------------------------------------
        # SVM
        # ------------------------------------------------------

        "SVM": {

            "model__C":

                loguniform(1e-3, 100),

            "model__gamma":

                ["scale", "auto"],

            "model__class_weight":

                [None, "balanced"]

        },

        # ------------------------------------------------------
        # MLP
        # ------------------------------------------------------

        "MLP": {

            "model__hidden_layer_sizes": [

                (64,),
                (128,),
                (64, 32),
                (128, 64)

            ],

            "model__alpha":

                loguniform(1e-5, 1e-1),

            "model__learning_rate_init":

                loguniform(1e-4, 1e-2),

            "model__batch_size":

                [16, 32, 64],

            "model__activation":

                ["relu", "tanh"]

        }

    }
    return PARAM_DISTRIBUTIONS


# ==========================================================
# Randomized Search Iterations
# ==========================================================

def get_search_iter():
    SEARCH_ITERATIONS = {

        "Logistic Regression": 15,

        "Decision Tree": 20,

        "Random Forest": 30,

        "SVM": 25,

        "MLP": 25,

        "XGBoost": 40,

        "LightGBM": 40,

        "CatBoost": 40

    }
    return SEARCH_ITERATIONS


# ==========================================================
# Optimization Metric
# ==========================================================

# def get_selection_metric
# SCORING = {

#     "Dataset1": "roc_auc",

#     "Dataset2": "average_precision"

# }