from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)

def get_grid_params(model_name):
    if model_name == 'RandomForest':
        return {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [1, 2, 4]
        }

    if model_name == 'LogisticRegression':
        return {
            'model__C': [0.01, 0.1, 1.0],
            'model__solver': ['liblinear']
        }

    if model_name == 'LinearSVM':
        return {
            'model__C': [0.1, 1, 10]
        }

    if model_name == 'MLP':
        return {
            'model__hidden_layer_sizes': [(50,), (100,)],
            'model__alpha': [0.0001, 0.001]
        }

    if model_name == 'HistGradientBoosting':
        return {
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [None, 10],
            'model__max_iter': [100, 200]
        }

    if model_name == 'XGBoost':
        return {
            'model__n_estimators': [200, 300],
            'model__max_depth': [4, 6],
            'model__learning_rate': [0.05, 0.1]
        }

    if model_name == 'LightGBM':
        return {
            'model__n_estimators': [200, 300],
            'model__num_leaves': [31, 63],
            'model__learning_rate': [0.05, 0.1]
        }

    return None


def perform_grid_search(name, model, X, y, groups=None, cv=5):
    param_grid = get_grid_params(name)
    if param_grid is None:
        return model

    print(f'Iniciando grid search para {name}...')

    cv_splitter = _build_cv_splitter(groups, cv)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=cv_splitter,
        n_jobs=1,
        verbose=0
    )
    
    if groups is not None:
        search.fit(X, y, groups=groups)
    else:
        search.fit(X, y)
        
    print('Melhores parâmetros:', search.best_params_)
    print('Melhor F1 (CV):', round(search.best_score_, 4))
    return search.best_estimator_


def _build_cv_splitter(groups, cv):
    if groups is not None:
        print(f'Usando GroupKFold com {len(set(groups))} grupos únicos')
        return GroupKFold(n_splits=cv)
    return StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)


def _get_prediction_probabilities(model, X_test):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(X_test)
    return model.predict(X_test)


def _print_evaluation_summary(y_test, y_pred, y_prob):
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    if len(set(y_test)) > 1:
        try:
            print('ROC AUC:', roc_auc_score(y_test, y_prob))
        except Exception:
            pass

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    print('Precision / Recall / F1 / Support for class 0:', precision[0], recall[0], f1[0], support[0])
    print('Precision / Recall / F1 / Support for class 1:', precision[1], recall[1], f1[1], support[1])


def evaluate_model(name, model, X_train, y_train, X_test, y_test, threshold=0.3, grid_search=False, groups=None):
    print(f"\n========================")
    print(f"Modelo: {name}")
    print(f"========================")

    if grid_search:
        model = perform_grid_search(name, model, X_train, y_train, groups=groups)

    model.fit(X_train, y_train)
    y_prob = _get_prediction_probabilities(model, X_test)
    y_pred = (y_prob > threshold).astype(int)

    _print_evaluation_summary(y_test, y_pred, y_prob)

    return model
