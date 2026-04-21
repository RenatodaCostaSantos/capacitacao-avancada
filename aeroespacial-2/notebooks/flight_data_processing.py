import numpy as np
import pandas as pd


def sanitize_flight_dataframe(df):
    df = df.copy()

    for col in df.columns:
        if col not in ['%time', 'failure']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_cols = [c for c in df.columns if c not in ['%time', 'failure']]
    if feature_cols:
        df[feature_cols] = df[feature_cols].interpolate(method='linear', limit_direction='forward')
        df[feature_cols] = df[feature_cols].ffill()
        df[feature_cols] = df[feature_cols].fillna(0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].clip(-1e4, 1e4)

    if 'failure' in df.columns:
        df['failure'] = pd.to_numeric(df['failure'], errors='coerce').fillna(0)
        df['failure'] = (df['failure'] > 0).astype(int)

    return df


def normalize_sources_to_flight_t0(source_dfs):
    valid_starts = []
    for df in source_dfs:
        if '%time' in df.columns:
            t = pd.to_numeric(df['%time'], errors='coerce').dropna()
            if len(t) > 0:
                valid_starts.append(float(t.min()))

    if len(valid_starts) == 0:
        return None

    t0 = float(np.min(valid_starts))
    normalized = []
    for df in source_dfs:
        df_local = df.copy()
        df_local['%time'] = (pd.to_numeric(df_local['%time'], errors='coerce') - t0) / 1e9
        df_local = df_local.dropna(subset=['%time']).sort_values('%time')
        normalized.append(df_local)

    return normalized
