import pandas as pd
import numpy as np


def _to_numeric_time(df):
    t = pd.to_numeric(df['%time'], errors='coerce')
    return t

def load_velocity(path, flight_id):
    csv = path / f"{flight_id}-mavros-local_position-velocity.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)

    for col in ['field.twist.linear.x', 'field.twist.linear.y', 'field.twist.linear.z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[
        'field.twist.linear.x',
        'field.twist.linear.y',
        'field.twist.linear.z'
    ])

    if df.empty:
        return None

    t = _to_numeric_time(df)
    data = {
        '%time': t,
        'vel_x': df['field.twist.linear.x'],
        'vel_y': df['field.twist.linear.y'],
        'vel_z': df['field.twist.linear.z'],
    }
    data['vel'] = np.sqrt(
        data['vel_x']**2 +
        data['vel_y']**2 +
        data['vel_z']**2
    )

    return pd.DataFrame(data)

def load_imu(path, flight_id):
    csv = path / f"{flight_id}-mavros-imu-data.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)
    t = _to_numeric_time(df)

    data = {'%time': t}

    # aceleração
    for axis in ['x', 'y', 'z']:
        col = f'field.linear_acceleration.{axis}'
        if col in df:
            data[f'acc_{axis}'] = pd.to_numeric(df[col], errors='coerce')

    # giroscópio
    for axis in ['x', 'y', 'z']:
        col = f'field.angular_velocity.{axis}'
        if col in df:
            data[f'gyro_{axis}'] = pd.to_numeric(df[col], errors='coerce')

    df_out = pd.DataFrame(data)

    df_out = df_out.ffill().bfill()

    # magnitude aceleração
    if all(c in df_out for c in ['acc_x', 'acc_y', 'acc_z']):
        df_out['acc_mag'] = np.linalg.norm(
            df_out[['acc_x', 'acc_y', 'acc_z']].values, axis=1
        )

    # magnitude gyro
    if all(c in df_out for c in ['gyro_x', 'gyro_y', 'gyro_z']):
        df_out['gyro_mag'] = np.linalg.norm(
            df_out[['gyro_x', 'gyro_y', 'gyro_z']].values, axis=1
        )

    return df_out

def load_rpy(path, flight_id):
    csv = path / f"{flight_id}-mavctrl-rpy.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)
    t = _to_numeric_time(df)

    col_map = {'roll': None, 'pitch': None, 'yaw': None}

    for col in df.columns:
        c = col.lower()
        if 'roll' in c or c.endswith('.x'):
            col_map['roll'] = col
        elif 'pitch' in c or c.endswith('.y'):
            col_map['pitch'] = col
        elif 'yaw' in c or c.endswith('.z'):
            col_map['yaw'] = col

    return t, df, col_map

def load_failure(path, flight_id):
    files = list(path.glob(f"{flight_id}-failure_status-*.csv"))

    if len(files) == 0:
        return None

    dfs = []

    for f in files:
        df = pd.read_csv(f)

        if 'field.data' not in df:
            continue

        df['field.data'] = pd.to_numeric(df['field.data'], errors='coerce')
        df = df.dropna(subset=['field.data'])

        t = _to_numeric_time(df)

        dfs.append(pd.DataFrame({
            '%time': t,
            'failure': df['field.data']
        }))

    if len(dfs) == 0:
        return None

    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.sort_values('%time')
    df_final = df_final.groupby('%time', as_index=False)['failure'].max()

    return df_final['%time'], df_final['failure']


def load_rc(path, flight_id):
    csv = path / f"{flight_id}-mavros-rc-out.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)
    t = _to_numeric_time(df)

    return t, df