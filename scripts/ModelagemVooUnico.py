from pathlib import Path
import pandas as pd
import os

# Define o caminho base como o diretório onde o script está, 
# e sobe um nível (.parent) para chegar na raiz do repositório
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_higienizada"

print(f"Buscando dados em: {DATA_ROOT}")

# Verifica se a pasta existe antes de continuar
if not DATA_ROOT.exists():
    # Caso a estrutura tenha uma subpasta 'processed_higienizado'
    DATA_ROOT = DATA_ROOT / "processed_higienizado"
    if not DATA_ROOT.exists():
        print(f"Erro: A pasta {DATA_ROOT} não foi encontrada.")
        # Lista o que existe na raiz para ajudar no debug
        print(f"Conteúdo na raiz: {os.listdir(BASE_DIR)}")
        exit()

# Selecionar apenas pastas de voo (ignorar _reports etc)
flights = [p for p in DATA_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")]

if not flights:
    print(f"Nenhuma pasta de voo encontrada em: {DATA_ROOT}")
    exit()

# Escolher o primeiro voo
flight = flights[0]
print(f"\nVoo selecionado: {flight.name}")

# Localizar arquivos usando rglob (recursivo) para garantir que encontre os CSVs
battery_files = list(flight.rglob("*battery*.csv"))
odom_files = list(flight.rglob("*odom*.csv"))

print("\nArquivos encontrados:")
print(f"Battery: {[f.name for f in battery_files]}")
print(f"Odom: {[f.name for f in odom_files]}")

# Carregar se as listas não estiverem vazias
if battery_files and odom_files:
    battery = pd.read_csv(battery_files[0])
    odom = pd.read_csv(odom_files[0])

    print("\n--- Colunas Carregadas ---")
    print(f"Bateria: {list(battery.columns)}")
    print(f"Odom: {list(odom.columns)}")
else:
    print("\nErro: Arquivos battery ou odom não encontrados dentro da pasta do voo.")

    #-------------------------------------------------------------------------------

    from pathlib import Path
import numpy as np
import pandas as pd
import os

BASE_DIR = Path(__file__).resolve().parent.parent
# Define a pasta onde os dados estão na sua branch 'Elaine'
DATA_ROOT = BASE_DIR / "data_higienizada"

# Seleciona a pasta específica do voo (ajustado para a estrutura do seu repositório)
flight_dirs = list(DATA_ROOT.rglob("carbonZ_2018-07-18-12-10-11_no_ground_truth"))

if not flight_dirs:
    print(f"Erro: Pasta do voo não encontrada em {DATA_ROOT}")
    # Lista o que foi encontrado para ajudar você a depurar
    print("Pastas disponíveis:", [p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
    exit()

flight_dir = flight_dirs[0]
# ---------------------------------------

# Localizar arquivos com rglob para garantir que encontre mesmo em subpastas
battery_files = list(flight_dir.rglob("*mavros-battery*.csv"))
odom_files = list(flight_dir.rglob("*mavros-local_position-odom*.csv"))

if not battery_files or not odom_files:
    print(f"Erro: Arquivos CSV não encontrados em {flight_dir}")
    exit()

battery = pd.read_csv(battery_files[0], low_memory=False)
odom = pd.read_csv(odom_files[0], low_memory=False)

def make_time_s(df, time_col="pcttime"):
    t = pd.to_numeric(df[time_col], errors="coerce")
    t_np = t.to_numpy()
    med = np.nanmedian(t_np) if np.isfinite(t_np).any() else np.nan

    if np.isfinite(med) and med > 1e12:
        t = t / 1e9  # ns -> s
    elif np.isfinite(med) and med > 1e9:
        t = t / 1e3  # ms -> s
    else:
        t = t.astype("float64")

    t0 = np.nanmin(t.to_numpy()) if np.isfinite(t.to_numpy()).any() else 0.0
    return (t - t0).astype("float64")

# Processamento dos dados
battery["time_s"] = make_time_s(battery, "pcttime")
odom["time_s"] = make_time_s(odom, "pcttime")

battery = battery.drop_duplicates(subset=["time_s"]).sort_values("time_s")
odom = odom.drop_duplicates(subset=["time_s"]).sort_values("time_s")

# Seleção de colunas
battery = battery[[
    "time_s", "field_voltage", "field_current", "field_percentage",
    "field_power_supply_status", "field_power_supply_health", "field_present",
]]

odom = odom[[
    "time_s", "field_pose_pose_position_x", "field_pose_pose_position_y",
    "field_pose_pose_position_z", "field_twist_twist_linear_x",
    "field_twist_twist_linear_y", "field_twist_twist_linear_z",
    "field_twist_twist_angular_x", "field_twist_twist_angular_y",
    "field_twist_twist_angular_z",
]]

# Renomeação
battery = battery.rename(columns={
    "field_voltage": "voltage_v",
    "field_current": "current_a",
    "field_percentage": "battery_pct",
    "field_power_supply_status": "battery_status",
    "field_power_supply_health": "battery_health",
    "field_present": "battery_present",
})

odom = odom.rename(columns={
    "field_pose_pose_position_x": "x_m",
    "field_pose_pose_position_y": "y_m",
    "field_pose_pose_position_z": "z_m",
    "field_twist_twist_linear_x": "vx_mps",
    "field_twist_twist_linear_y": "vy_mps",
    "field_twist_twist_linear_z": "vz_mps",
    "field_twist_twist_angular_x": "wx_rps",
    "field_twist_twist_angular_y": "wy_rps",
    "field_twist_twist_angular_z": "wz_rps",
})

# Consolidação
consolidated = pd.merge_asof(
    odom.sort_values("time_s"),
    battery.sort_values("time_s"),
    on="time_s",
    direction="nearest",
    tolerance=0.25
)

consolidated["speed_mps"] = np.sqrt(
    consolidated["vx_mps"]**2 + consolidated["vy_mps"]**2 + consolidated["vz_mps"]**2
)
consolidated["power_w"] = consolidated["voltage_v"] * consolidated["current_a"]

# Salvar o resultado na mesma pasta do voo dentro do GitHub
out_path = flight_dir / "flight_consolidated.csv"
consolidated.to_csv(out_path, index=False)

print("--- Execução Finalizada ---")
print("Salvo em:", out_path)
print("Linhas:", consolidated.shape[0], "| Colunas:", consolidated.shape[1])

#-------------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
import numpy as np
import os

# Define a raiz do repositório a partir da localização deste script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_higienizada"

# Localiza a pasta do voo específica (usando rglob para garantir que encontre)
flight_dirs = list(DATA_ROOT.rglob("carbonZ_2018-07-18-12-10-11_no_ground_truth"))

if not flight_dirs:
    print(f"Erro: Pasta do voo não encontrada em {DATA_ROOT}")
    exit()

flight_dir = flight_dirs[0]
input_file = flight_dir / "flight_consolidated.csv"

# Verificar se o arquivo gerado na etapa anterior existe
if not input_file.exists():
    print(f"Erro: Arquivo {input_file.name} não encontrado. Rode o script de consolidação primeiro.")
    exit()
# -------------------------------------------

# Carregar os dados consolidados
df = pd.read_csv(input_file)

# Evitar divisão por zero ou valores negativos irreais na potência
df["power_w"] = df["power_w"].replace(0, np.nan)

# Cálculo da eficiência instantânea (Razão entre velocidade de subida e potência)
# Adicionado tratamento para evitar valores infinitos se power_w for muito pequeno
df["motor_efficiency"] = df["vz_mps"] / df["power_w"]

# Salvar o novo arquivo com a coluna de eficiência na mesma pasta do GitHub
output_file = flight_dir / "flight_consolidated_with_efficiency.csv"
df.to_csv(output_file, index=False)

print(f"--- Processamento Concluído ---")
print(f"Arquivo salvo em: {output_file}")
print(f"\nResumo da Eficiência do Motor:")
print(df["motor_efficiency"].describe())

# Opcional: Mostrar as primeiras linhas para conferência
print("\nPrimeiras linhas do resultado:")
print(df[["time_s", "vz_mps", "power_w", "motor_efficiency"]].head())

#--------------------------------------------

print(df[["voltage_v","current_a","power_w"]].describe())

#--------------------------------------------

print(df[["voltage_v","current_a","power_w"]].head(10))

#----------------------------------------------------

import numpy as np

df["effort"] = np.sqrt(
    df["vx_mps"]**2 +
    df["vy_mps"]**2 +
    df["vz_mps"]**2
)

print(df["effort"].describe())

#----------------------------------------------------

print(df["vz_mps"].describe())

#---------------------------------------------------

df["speed_horizontal"] = np.sqrt(
    df["vx_mps"]**2 + df["vy_mps"]**2
)

print(df["speed_horizontal"].describe())

#----------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# BASE_DIR aponta para a raiz do repositório (/workspaces/capacitacao-avancada)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_higienizada"

# Localiza a pasta do voo dentro do repositório
flight_dirs = list(DATA_ROOT.rglob("carbonZ_2018-07-18-12-10-11_no_ground_truth"))

if not flight_dirs:
    print(f"Erro: Pasta do voo não encontrada em {DATA_ROOT}")
    exit()

flight_path = flight_dirs[0]
file_path = flight_path / "flight_consolidated.csv"

# Verificar se o arquivo existe antes de ler
if not file_path.exists():
    print(f"Erro: Arquivo {file_path.name} não encontrado em {flight_path}")
    exit()

# Leitura do arquivo
df = pd.read_csv(file_path)

# Verificar colunas
required = ["vx_mps", "vy_mps", "vz_mps"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"A coluna {col} não está presente no arquivo.")

# Cálculos
df["speed_horizontal"] = np.sqrt(df["vx_mps"]**2 + df["vy_mps"]**2)
data = df[["speed_horizontal", "vz_mps"]].replace([np.inf, -np.inf], np.nan).dropna()

# --- GRÁFICOS ---

plt.figure(figsize=(7,5))
plt.scatter(data["speed_horizontal"], data["vz_mps"], s=10, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Velocidade horizontal (m/s)")
plt.ylabel("Velocidade vertical vz (m/s)")
plt.title("Relação entre velocidade horizontal e vertical")
plt.grid(True)
plt.savefig(flight_path / "grafico_scatter.png") # Salva na pasta do voo

plt.figure(figsize=(7,5))
plt.hexbin(data["speed_horizontal"], data["vz_mps"], gridsize=35, mincnt=1, cmap='viridis')
plt.axhline(0, color='white', linestyle='--')
plt.xlabel("Velocidade horizontal (m/s)")
plt.ylabel("Velocidade vertical vz (m/s)")
plt.title("Distribuição de vz")
plt.grid(True)
plt.colorbar(label='Contagem')
plt.savefig(flight_path / "grafico_densidade.png") # Salva na pasta do voo

print(f"Sucesso! Gráficos salvos em: {flight_path}")

#--------------------------------------------------

if {"wx_rps","wy_rps","wz_rps"}.issubset(df.columns):
    df["w_norm"] = np.sqrt(df["wx_rps"]**2 + df["wy_rps"]**2 + df["wz_rps"]**2)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["w_norm"], df["vz_mps"], s=10, alpha=0.6)
    plt.axhline(0.0)
    plt.xlabel("w_norm (rad/s)")
    plt.ylabel("vz_mps (m/s)")
    plt.title("vz vs norma da velocidade angular (estabilidade)")
    plt.grid(True)
    plt.show()

    #-----------------------------------------------

    plt.figure(figsize=(10,4))
plt.plot(df["time_s"], df["vx_mps"], label="vx")
plt.plot(df["time_s"], df["vy_mps"], label="vy")
plt.plot(df["time_s"], df["vz_mps"], label="vz")

plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.title("Componentes da velocidade ao longo do voo")
plt.legend()
plt.grid(True)
plt.show()

#------------------------------------------------

plt.figure(figsize=(6,6))
plt.plot(df["x_m"], df["y_m"])

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajetória do voo no plano XY")
plt.axis("equal")
plt.grid(True)
plt.show()

#----------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Pega a raiz do repositório subindo um nível da pasta 'scripts'
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_higienizada"

# Localiza a pasta do voo dentro da estrutura da branch 'Elaine'
flight_dirs = list(DATA_ROOT.rglob("carbonZ_2018-07-18-12-10-11_no_ground_truth"))

if not flight_dirs:
    print(f"Erro: Pasta do voo não encontrada em {DATA_ROOT}")
    exit()

flight_dir = flight_dirs[0]
csv_path = flight_dir / "flight_consolidated.csv"

# Verifica se o arquivo gerado anteriormente existe
if not csv_path.exists():
    print(f"Erro: Arquivo {csv_path.name} não encontrado. Execute o script de consolidação primeiro.")
    exit()
# -----------------------------------------------

df = pd.read_csv(csv_path)

# Verificar colunas necessárias
for col in ["time_s", "vx_mps", "vy_mps", "vz_mps"]:
    if col not in df.columns:
        raise ValueError(f"Coluna ausente: {col}")

# Conversão para numpy para cálculos matemáticos
t = df["time_s"].to_numpy(dtype=float)
vx = df["vx_mps"].to_numpy(dtype=float)
vy = df["vy_mps"].to_numpy(dtype=float)

eps = 1e-9

# Cálculo da velocidade horizontal e direção
speed_h = np.hypot(vx, vy)
ux = vx / (speed_h + eps)
uy = vy / (speed_h + eps)

# Vetor perpendicular e deriva lateral (sideslip proxy)
nx = -uy
ny = ux
v_lat = vx * nx + vy * ny

df["speed_horizontal"] = speed_h
df["v_lateral_mps"] = v_lat

# Aceleração e curvatura (yaw rate proxy)
dt = np.gradient(t)
ax = np.gradient(vx) / (dt + eps)
ay = np.gradient(vy) / (dt + eps)
yaw_proxy = (vx * ay - vy * ax) / (speed_h**2 + eps)
df["yaw_rate_proxy_rps"] = yaw_proxy

# Suavização para o gráfico
df["v_lateral_smooth"] = df["v_lateral_mps"].rolling(25, center=True, min_periods=1).median()

# Limpeza de dados para plotagem
plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "speed_horizontal", "v_lateral_smooth", "yaw_rate_proxy_rps"])
plot_df = plot_df[plot_df["speed_horizontal"] > 1.0]

# --- GERAÇÃO E SALVAMENTO DE GRÁFICOS NO GITHUB ---

# 1. Deriva lateral ao longo do tempo
plt.figure(figsize=(10,4))
plt.plot(plot_df["time_s"], plot_df["v_lateral_smooth"])
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Tempo (s)")
plt.ylabel("Deriva lateral (m/s)")
plt.title("Deriva lateral ao longo do voo")
plt.grid(True)
plt.savefig(flight_dir / "analise_deriva_temporal.png")

# 2. Deriva vs velocidade horizontal (Densidade)
plt.figure(figsize=(7,5))
plt.hexbin(plot_df["speed_horizontal"], plot_df["v_lateral_smooth"], gridsize=40, mincnt=1, cmap='magma')
plt.axhline(0, color='white', linestyle='--')
plt.xlabel("Velocidade horizontal (m/s)")
plt.ylabel("Deriva lateral (m/s)")
plt.title("Deriva lateral vs velocidade horizontal")
plt.colorbar(label='Densidade de Pontos')
plt.grid(True)
plt.savefig(flight_dir / "analise_deriva_hexbin.png")

# 3. Curvatura da trajetória ao longo do tempo
plt.figure(figsize=(10,4))
plt.plot(plot_df["time_s"], plot_df["yaw_rate_proxy_rps"], color='green')
plt.axhline(0, color='black', lw=1)
plt.xlabel("Tempo (s)")
plt.ylabel("Yaw rate proxy (rad/s)")
plt.title("Curvatura da trajetória (Proxy de Yaw Rate)")
plt.grid(True)
plt.savefig(flight_dir / "analise_curvatura_trajetoria.png")

print(f"\n--- Processamento de Dinâmica Finalizado ---")
print(f"Gráficos salvos na pasta: {flight_dir}")

# Mostra no Codespace (se houver suporte visual)
plt.show()

#-----------------------------------------------------------

import numpy as np
import pandas as pd

# precisa existir no df
for col in ["time_s", "yaw_rate_proxy_rps"]:
    if col not in df.columns:
        raise ValueError(f"Coluna ausente: {col}")

threshold = 6.0  # 5 a 8 costuma funcionar

tmp = df[["time_s", "yaw_rate_proxy_rps"]].copy()
tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

x = tmp["yaw_rate_proxy_rps"].to_numpy(dtype=float)

med = np.median(x)
mad = np.median(np.abs(x - med)) + 1e-12

tmp["z_robust"] = 0.6745 * (tmp["yaw_rate_proxy_rps"] - med) / mad
tmp["anomaly_flag"] = tmp["z_robust"].abs() > threshold

print("Pontos sinalizados:", int(tmp["anomaly_flag"].sum()), "de", len(tmp))
print("Taxa:", float(tmp["anomaly_flag"].mean()))

print(tmp.loc[tmp["anomaly_flag"], ["time_s", "yaw_rate_proxy_rps", "z_robust"]].head(30))

#-------------------------------------------------------

import numpy as np

z_abs_max = float(tmp["z_robust"].abs().max())
print("max |z| =", z_abs_max)

qs = tmp["z_robust"].abs().quantile([0.90, 0.95, 0.99, 0.995]).to_dict()
print("quantis |z|:", {k: float(v) for k, v in qs.items()})

#----------------------------------------------------------

threshold = 3.0

tmp["anomaly_flag"] = tmp["z_robust"].abs() > threshold

print("Pontos sinalizados:", int(tmp["anomaly_flag"].sum()), "de", len(tmp))
print("Taxa:", float(tmp["anomaly_flag"].mean()))

print(tmp.loc[tmp["anomaly_flag"],
              ["time_s","yaw_rate_proxy_rps","z_robust"]].head(20))

#------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(tmp["time_s"], tmp["yaw_rate_proxy_rps"])

an = tmp[tmp["anomaly_flag"]]

plt.scatter(an["time_s"], an["yaw_rate_proxy_rps"], color="red")

plt.xlabel("Tempo (s)")
plt.ylabel("yaw_rate_proxy")
plt.title("Eventos raros na curvatura da trajetória")
plt.grid(True)
plt.show()

#--------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Pega a raiz do repositório (/workspaces/capacitacao-avancada)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_higienizada"

# Localiza a pasta do voo dentro do repositório usando busca recursiva
flight_dirs = list(DATA_ROOT.rglob("carbonZ_2018-07-18-12-10-11_no_ground_truth"))

if not flight_dirs:
    print(f"Erro: Pasta do voo não encontrada em {DATA_ROOT}")
    exit()

flight_path = flight_dirs[0]
csv_path = flight_path / "flight_consolidated.csv"

# Verifica se o arquivo existe
if not csv_path.exists():
    print(f"Erro: Arquivo {csv_path.name} não encontrado. Execute o script de consolidação primeiro.")
    exit()
# ---------------------------------------

df = pd.read_csv(csv_path)

# Verificar colunas necessárias
for col in ["time_s", "vx_mps", "vy_mps", "x_m", "y_m"]:
    if col not in df.columns:
        raise ValueError(f"Coluna ausente: {col}")

t  = df["time_s"].to_numpy()
vx = df["vx_mps"].to_numpy()
vy = df["vy_mps"].to_numpy()

eps = 1e-9
dt = np.gradient(t)

# Cálculo de aceleração e Yaw Rate Proxy
ax = np.gradient(vx) / (dt + eps)
ay = np.gradient(vy) / (dt + eps)
speed2 = vx**2 + vy**2
yaw_rate = (vx * ay - vy * ax) / (speed2 + eps)

df["yaw_rate_proxy_rps"] = yaw_rate

# Detecção de anomalias (Z-score robusto)
x_clean = df["yaw_rate_proxy_rps"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
med = np.median(x_clean)
mad = np.median(np.abs(x_clean - med)) + 1e-12

df["z_robust"] = 0.6745 * (df["yaw_rate_proxy_rps"] - med) / mad
threshold = 3.0
df["anomaly"] = df["z_robust"].abs() > threshold

print("Pontos sinalizados (anomalias):", df["anomaly"].sum())

# --- GRÁFICOS ---
plt.figure(figsize=(8, 8))
plt.plot(df["x_m"], df["y_m"], label="Trajetória Nominal", alpha=0.7)

# Destacar anomalias em vermelho
anomalies = df[df["anomaly"]]
plt.scatter(anomalies["x_m"], anomalies["y_m"], color="red", label="Eventos Raros", zorder=5)

plt.xlabel("Posição X (m)")
plt.ylabel("Posição Y (m)")
plt.title("Trajetória do Voo com Detecção de Eventos Raros")
plt.legend()
plt.grid(True)
plt.axis("equal")

# Salvar o gráfico na pasta do voo para visualização no GitHub
plt.savefig(flight_path / "trajetoria_anomalias.png")
print(f"Gráfico salvo em: {flight_path}/trajetoria_anomalias.png")

plt.show()