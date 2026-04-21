import numpy as np
import pandas as pd

SAMPLE_RATE = 4.0
SAMPLE_DT = 1.0 / SAMPLE_RATE
MERGE_TOLERANCE = 0.35

FEATURE_COLUMNS = [
    'vel', 'vel_x', 'vel_y', 'vel_z',
    'roll', 'pitch', 'yaw',
    'acc_x', 'acc_y', 'acc_z', 'acc_mag', 'acc_xy',
    'gyro_x', 'gyro_y', 'gyro_z', 'gyro_mag', 'gyro_xy',
    'rc0', 'rc1', 'rc2', 'rc3',
    'vel_diff',
    'roll_diff', 'pitch_diff', 'yaw_diff',
    'acc_x_diff', 'acc_y_diff', 'acc_z_diff',
    'gyro_x_diff', 'gyro_y_diff', 'gyro_z_diff',
    'rc1_diff', 'rc2_diff', 'rc3_diff'
]


def create_time_grid(dfs, sample_rate=SAMPLE_RATE):
    end_times = [
        df['%time'].max()
        for df in dfs
        if '%time' in df.columns and not df['%time'].isna().all()
    ]

    if len(end_times) == 0:
        return None

    max_time = float(np.nanmax(end_times))
    return np.arange(0.0, max_time + SAMPLE_DT / 2, SAMPLE_DT)


def merge_to_grid(df_final, df_source, failure=False):
    df_source = df_source.copy()
    df_source['%time'] = pd.to_numeric(df_source['%time'], errors='coerce')
    df_source = df_source.sort_values('%time').dropna(subset=['%time'])

    direction = 'backward'
    return pd.merge_asof(
        df_final,
        df_source,
        on='%time',
        direction=direction,
        tolerance=MERGE_TOLERANCE
    )


def derive_features(df):
    if all(c in df for c in ['acc_x', 'acc_y']):
        df['acc_xy'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2)
    if all(c in df for c in ['gyro_x', 'gyro_y']):
        df['gyro_xy'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2)

    # rc0 is typically constant (throttle); skip rc0_diff (zero variance in practice).
    for col in [
        'vel', 'roll', 'pitch', 'yaw',
        'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'rc1', 'rc2', 'rc3'
    ]:
        if col in df:
            df[f'{col}_diff'] = df[col].diff().fillna(0)

    return df
