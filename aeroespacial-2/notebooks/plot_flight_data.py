import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from dataset import prepare_flight_data
from features import FEATURE_COLUMNS

ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR / "../data/processed"
OUTPUT_DIR = ROOT_DIR / "plots_individuais"
OUTPUT_DIR.mkdir(exist_ok=True)


def _create_figure(flight_id):
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(f"Análise de Telemetria: {flight_id}", fontsize=14)
    return fig, axes


def _plot_velocity(ax, t, df):
    ax.plot(t, df['vel'], label='Velocidade (m/s)')
    ax.set_title("Velocidade")
    ax.set_ylabel("m/s")
    ax.grid(True)
    ax.legend()


def _plot_attitude(ax, t, df):
    ax.plot(t, df['roll'], label='Roll')
    ax.plot(t, df['pitch'], label='Pitch')
    ax.plot(t, df['yaw'], label='Yaw')
    ax.set_title("Atitude (Roll / Pitch / Yaw)")
    ax.set_ylabel("rad")
    ax.grid(True)
    ax.legend()


def _plot_rc(ax, t, df):
    for i in range(4):
        rc_col = f'rc{i}'
        if rc_col in df:
            ax.plot(t, df[rc_col], label=f'Ch{i}')
    ax.set_title("RC Out")
    ax.set_ylabel("PWM")
    ax.grid(True)
    ax.legend(ncol=4)


def _plot_failure(ax, t, df):
    ax.step(t, df['failure'], where='post', color='red', label='Failure')
    ax.set_title("Failure Status")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("0/1")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()


def _mark_failure_onset(axes, df):
    fail_idx = df.index[df['failure'] == 1]
    if len(fail_idx) == 0:
        return
    t_onset = float(df.loc[fail_idx[0], 't'])
    for ax in axes:
        ax.axvline(x=t_onset, color='red', linestyle='--', alpha=0.4)


def _save_figure(fig, flight_id):
    save_path = OUTPUT_DIR / f"analise_{flight_id}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"{flight_id}: plot salvo em {save_path}")


def plot_confusion_matrix(cm, labels, title, save_path, figsize=(6, 5)):
    """Salva heatmap da matriz de confusão (cm: array 2x2 ou lista aninhada)."""
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title, ylabel='Verdadeiro', xlabel='Predito')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]) if cm[i, j] == int(cm[i, j]) else f'{cm[i, j]:.1f}',
                    ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _save_correlation_heatmap(df, feature_cols, title, save_path, figsize=(14, 12)):
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        raise ValueError("Precisa de pelo menos 2 colunas para correlação")
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=figsize)
    try:
        import seaborn as sns
        sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    except Exception:
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.figure.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_yticklabels(cols)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_feature_importance(feature_names, importances, title, save_path, top_n=20, figsize=(10, 6)):
    im = np.asarray(importances, dtype=float).flatten()
    names = list(feature_names)
    if len(names) != len(im):
        raise ValueError("feature_names e importances devem ter o mesmo tamanho")
    order = np.argsort(np.abs(im))[::-1][:top_n]
    im = im[order]
    names = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, im, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importância")
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_correlation(flight_name=None, save_path=None, max_rows_per_flight=None):
    """
    Única entrada pública para heatmap de correlação das FEATURE_COLUMNS.
    flight_name=None → todos os voos; caso contrário, nome da pasta do voo.
    """
    if flight_name is None:
        flight_paths = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
        frames = []
        for p in flight_paths:
            df = prepare_flight_data(p)
            if df is None or df.empty:
                continue
            sub = df[FEATURE_COLUMNS]
            if max_rows_per_flight is not None and len(sub) > max_rows_per_flight:
                sub = sub.sample(n=max_rows_per_flight, random_state=42)
            frames.append(sub)
        if len(frames) == 0:
            print("Nenhum dado para correlação")
            return
        all_df = pd.concat(frames, ignore_index=True)
        if save_path is None:
            save_path = OUTPUT_DIR / "correlacao_features_todos_voos.png"
        title = "Correlação entre features (todos os voos)"
    else:
        p = BASE_DIR / flight_name
        if not p.is_dir():
            print(f"Voo não encontrado: {flight_name}")
            return
        df = prepare_flight_data(p)
        if df is None or df.empty:
            print(f"{flight_name}: sem dados")
            return
        all_df = df[FEATURE_COLUMNS]
        if save_path is None:
            save_path = OUTPUT_DIR / f"correlacao_{flight_name}.png"
        title = f"Correlação entre features — {flight_name}"

    _save_correlation_heatmap(all_df, FEATURE_COLUMNS, title, save_path)
    print(f"Correlação salva em {save_path}")


def plot_flight_data(flight_folder):
    path = Path(flight_folder)
    flight_id = path.name

    try:
        df = prepare_flight_data(path)
        if df is None or df.empty:
            print(f"{flight_id}: sem dados válidos após processamento")
            return

        t = df['t']
        fig, (ax1, ax2, ax3, ax4) = _create_figure(flight_id)
        _plot_velocity(ax1, t, df)
        _plot_attitude(ax2, t, df)
        _plot_rc(ax3, t, df)
        _plot_failure(ax4, t, df)
        _mark_failure_onset((ax1, ax2, ax3, ax4), df)
        _save_figure(fig, flight_id)

    except Exception as e:
        print(f"Erro em {flight_id}: {e}")


def main():
    import sys

    argv = sys.argv[1:]

    if len(argv) > 0 and argv[0] == "diagnostics":
        plot_correlation(flight_name=None)
        print("Dica: use plot_confusion_matrix(cm, ...) após treino, ou importe deste módulo.")
        return

    if len(argv) > 1 and argv[0] == "correlation":
        plot_correlation(flight_name=argv[1])
        return

    flight_paths = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])

    if len(argv) > 0:
        flight_name = argv[0]
        selected = [p for p in flight_paths if p.name == flight_name]
        if len(selected) == 0:
            print(f"Voo não encontrado: {flight_name}")
            print(f"Total de voos disponíveis: {len(flight_paths)}")
            print("Comandos extras: diagnostics | correlation <nome_voo>")
            return
        flight_paths = selected
        print(f"Plotando voo específico: {flight_name}\n")
    else:
        print(f"Total de voos encontrados: {len(flight_paths)}\n")

    for flight_path in flight_paths:
        plot_flight_data(flight_path)


if __name__ == "__main__":
    main()
