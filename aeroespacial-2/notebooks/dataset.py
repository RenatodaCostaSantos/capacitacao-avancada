import pandas as pd
import numpy as np
from loading_dataset import load_velocity, load_rpy, load_failure, load_imu, load_rc
from features import FEATURE_COLUMNS, create_time_grid, merge_to_grid, derive_features
from flight_data_processing import sanitize_flight_dataframe, normalize_sources_to_flight_t0

MIN_FLIGHT_LENGTH = 30

def _merge_flight_data(path, flight_id):
    dfs = []

    vel = load_velocity(path, flight_id)
    if vel is not None:
        dfs.append(vel)

    imu = load_imu(path, flight_id)
    if imu is not None and not imu.empty:
        dfs.append(imu)

    rpy = load_rpy(path, flight_id)
    if rpy is not None:
        t, df_rpy, col_map = rpy
        data = {'%time': t}
        for k, col in col_map.items():
            if col is not None:
                data[k] = pd.to_numeric(df_rpy[col], errors='coerce')
        dfs.append(pd.DataFrame(data))

    rc = load_rc(path, flight_id)
    if rc is not None:
        t, df_rc = rc
        data = {'%time': t}
        for i in range(4):
            col = f'field.channels{i}'
            if col in df_rc:
                data[f'rc{i}'] = pd.to_numeric(df_rc[col], errors='coerce')
        dfs.append(pd.DataFrame(data))

    fail = load_failure(path, flight_id)
    if fail is not None:
        t, failure = fail
        dfs.append(pd.DataFrame({'%time': t, 'failure': failure}))

    return dfs


def build_timeseries_dataset(path):
    flight_id = path.name
    dfs = _merge_flight_data(path, flight_id)

    if len(dfs) == 0:
        return None

    dfs = normalize_sources_to_flight_t0(dfs)
    if dfs is None:
        return None

    time_grid = create_time_grid(dfs)
    if time_grid is None or len(time_grid) == 0:
        return None

    df_final = pd.DataFrame({'%time': time_grid})
    for df in dfs:
        is_failure = 'failure' in df.columns
        df_final = merge_to_grid(df_final, df, failure=is_failure)

    df_final['t'] = df_final['%time'] - df_final['%time'].iloc[0]

    if 'failure' not in df_final.columns:
        df_final['failure'] = 0
    else:
        df_final['failure'] = pd.to_numeric(df_final['failure'], errors='coerce').fillna(0)

    df_final['failure'] = (df_final['failure'] > 0).astype(int)

    for col in FEATURE_COLUMNS:
        if col not in df_final.columns:
            df_final[col] = 0

    df_final = sanitize_flight_dataframe(df_final)
    df_final = derive_features(df_final)
    df_final = sanitize_flight_dataframe(df_final)
    df_final = df_final[['%time', 't'] + FEATURE_COLUMNS + ['failure']]

    return df_final


def prepare_flight_data(path):
    df = build_timeseries_dataset(path)
    if df is None or len(df) < MIN_FLIGHT_LENGTH:
        return None
    return df.ffill()


def build_sequences(df, window_size=20):
    X = []
    y = []

    features = df[FEATURE_COLUMNS]
    target = df['failure']

    for i in range(len(df) - window_size):
        X.append(features.iloc[i:i + window_size].values)
        y.append(target.iloc[i + window_size])

    if len(X) == 0:
        return None, None

    return np.array(X), np.array(y)


def build_full_dataset(paths, window_size=20):
    all_X = []
    all_y = []

    for path in paths:
        df = prepare_flight_data(path)
        if df is None:
            continue

        X, y = build_sequences(df, window_size)
        if X is None:
            continue

        all_X.append(X)
        all_y.append(y)

    if len(all_X) == 0:
        raise ValueError("Nenhum dado válido")

    return np.concatenate(all_X), np.concatenate(all_y)


def split_by_flight(paths, train_ratio=0.8, random_state=42):
    paths = sorted(paths)
    rng = np.random.RandomState(random_state)
    rng.shuffle(paths)
    n = int(len(paths) * train_ratio)
    return paths[:n], paths[n:]
