import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from dataset import build_full_dataset, split_by_flight, prepare_flight_data, build_sequences
from evaluation import evaluate_model, _get_prediction_probabilities
from models import get_models


# ==============================
# BALANCEAMENTO
# ==============================
def balance_training_data(X, y, groups=None, strategy='oversample', random_state=42):
    if strategy != 'oversample' or groups is None:
        return X, y, groups

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or counts.min() == counts.max():
        return X, y, groups

    majority_label = labels[np.argmax(counts)]
    minority_label = labels[np.argmin(counts)]

    idx_majority = np.where(y == majority_label)[0]
    idx_minority = np.where(y == minority_label)[0]

    resampled_minority_idx = resample(
        idx_minority,
        replace=True,
        n_samples=len(idx_majority),
        random_state=random_state
    )

    final_indices = np.concatenate([idx_majority, resampled_minority_idx])
    perm = np.random.RandomState(random_state).permutation(len(final_indices))
    final_indices = final_indices[perm]

    return X[final_indices], y[final_indices], groups[final_indices]


# ==============================
# LIMPEZA NUMÉRICA
# ==============================
def _clean_data(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)

    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])

    X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
    X = np.clip(X, -1e4, 1e4)
    return X.astype(np.float32)


# ==============================
# SCALING
# ==============================
def _scale_temporal_data(X_train, X_test):
    n_samples, timesteps, n_features = X_train.shape

    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(n_samples, timesteps * n_features)
    X_test = X_test_2d.reshape(X_test.shape[0], timesteps * n_features)

    return X_train, X_test


def _build_training_groups(train_pastas, window_size):
    flight_indices = []
    for flight_idx, pasta in enumerate(train_pastas):
        df = prepare_flight_data(pasta)
        if df is None:
            continue
        X_flight, _ = build_sequences(df, window_size)
        if X_flight is not None:
            flight_indices.extend([flight_idx] * X_flight.shape[0])
    return np.array(flight_indices)


def _finalize_features(X_train, X_test):
    X_train = _clean_data(X_train)
    X_test = _clean_data(X_test)
    return _scale_temporal_data(X_train, X_test)

# ==============================
# PIPELINE
# ==============================
def make_pipeline(model, use_scaling):
    steps = []

    if use_scaling:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))
    return Pipeline(steps)


# ==============================
# PREPARAÇÃO DOS DADOS
# ==============================
def _prepare_datasets(flight_paths, window_size, balance_method, random_state):
    train_flight_paths, test_flight_paths = split_by_flight(flight_paths, random_state=random_state)

    print(f"Voos treino: {len(train_flight_paths)}")
    print(f"Voos teste: {len(test_flight_paths)}")

    X_train, y_train = build_full_dataset(train_flight_paths, window_size)
    X_test, y_test = build_full_dataset(test_flight_paths, window_size)

    print("\nTreino shape:", X_train.shape)
    print("Teste shape:", X_test.shape)

    flight_indices = _build_training_groups(train_flight_paths, window_size)

    X_train, y_train, groups_train = balance_training_data(
        X_train,
        y_train,
        groups=flight_indices,
        strategy=balance_method,
        random_state=random_state
    )

    print('Treino balanceado:', np.bincount(y_train))

    X_train, X_test = _finalize_features(X_train, X_test)

    return X_train, X_test, y_train, y_test, groups_train

# ==============================
# TREINO + RELATÓRIO
# ==============================
def train_and_evaluate_report(
    flight_paths,
    window_size=20,
    threshold=0.3,
    grid_search=False,
    balance_method='oversample',
    random_state=42,
    report_file=None
):
    X_train, X_test, y_train, y_test, groups_train = _prepare_datasets(
        flight_paths,
        window_size,
        balance_method,
        random_state
    )

    models = get_models()
    trained_models = {}
    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TREINANDO MODELO: {name}")
        print(f"{'='*50}")

        use_scaling = False

        pipeline = make_pipeline(model, use_scaling)
        trained_model = evaluate_model(
            name,
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            threshold,
            grid_search,
            groups=groups_train if grid_search else None
        )

        trained_models[name] = trained_model

        y_prob = _get_prediction_probabilities(trained_model, X_test)
        y_pred = (y_prob > threshold).astype(int)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1], zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'precision_class_0': precision[0],
            'recall_class_0': recall[0],
            'f1_class_0': f1[0],
            'support_class_0': support[0],
            'precision_class_1': precision[1],
            'recall_class_1': recall[1],
            'f1_class_1': f1[1],
            'support_class_1': support[1],
            'roc_auc': roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else None,
            'confusion_matrix': cm
        }

    if report_file:
        _generate_report(results, report_file, flight_paths, window_size, threshold, grid_search)

    _save_confusion_matrix_plots(results)

    return trained_models, results


def _save_confusion_matrix_plots(results):
    from plot_flight_data import plot_confusion_matrix

    out_dir = Path(__file__).resolve().parent / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Normal (0)", "Falha (1)"]
    for name, metrics in results.items():
        cm = np.asarray(metrics["confusion_matrix"], dtype=float)
        safe = "".join(c if c.isalnum() else "_" for c in name)
        path = out_dir / f"confusion_matrix_{safe}.png"
        plot_confusion_matrix(cm, labels, f"Matriz de confusão — {name}", path)
    print(f"Matrizes de confusão (PNG) em: {out_dir}")


# ==============================
# RELATÓRIO
# ==============================
def _generate_report(results, report_file, flight_paths, window_size, threshold, grid_search):
    import datetime

    with open(report_file, 'w') as f:
        f.write("RELATÓRIO DE TREINAMENTO DE MODELOS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data/Hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de voos: {len(flight_paths)}\n")
        f.write(f"Tamanho da janela: {window_size}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Grid Search: {'Sim' if grid_search else 'Não'}\n\n")

        f.write("RESULTADOS POR MODELO:\n")
        f.write("-" * 50 + "\n")

        for model_name, metrics in results.items():
            f.write(f"\nModelo: {model_name}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Classe 0 (Normal):\n")
            f.write(f"  Precision: {metrics['precision_class_0']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_class_0']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_class_0']:.4f}\n")
            f.write(f"  Support:   {metrics['support_class_0']}\n")
            f.write(f"Classe 1 (Falha):\n")
            f.write(f"  Precision: {metrics['precision_class_1']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_class_1']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_class_1']:.4f}\n")
            f.write(f"  Support:   {metrics['support_class_1']}\n")
            if metrics['roc_auc'] is not None:
                f.write(f"ROC AUC:     {metrics['roc_auc']:.4f}\n")
            f.write("Matriz de Confusão:\n")
            f.write(f"  [[{metrics['confusion_matrix'][0][0]}, {metrics['confusion_matrix'][0][1]}],\n")
            f.write(f"   [{metrics['confusion_matrix'][1][0]}, {metrics['confusion_matrix'][1][1]}]]\n")
            f.write("\n")