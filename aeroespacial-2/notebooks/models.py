from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

def get_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=1,
            class_weight={0: 1, 1: 5},
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=10,
            learning_rate=0.05,
            max_iter=200,
            random_state=42
        ),

        "LogisticRegression": LogisticRegression(
            class_weight='balanced',
            max_iter=5000,
            C=0.1,
            solver='liblinear',
            penalty='l2',
            random_state=42
        ),
        "LinearSVM": LinearSVC(
            class_weight='balanced',
            max_iter=2000
        ),

        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            learning_rate_init=0.001,
            random_state=42
        )
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
            verbosity=-1
        )
    except Exception:
        pass

    return models