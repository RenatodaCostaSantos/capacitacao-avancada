from __future__ import annotations
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Metadados comuns que aparecem em quase todos os arquivos
COMMON_COLUMNS = [
    ("%time", "Timestamp do registro no CSV (nanossegundos). Instante da gravação no arquivo."),
    ("field.header.seq", "Número sequencial da mensagem ROS. Usado para detectar perda de pacotes."),
    ("field.header.stamp", "Timestamp do ROS (segundos/nanossegundos). Quando a mensagem foi gerada no sistema."),
    ("field.header.frame_id", "Frame de referência (ex: base_link, map, local_origin)."),
]

# 2) Colunas específicas de cada tópico/arquivo
FILES = {
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-imu-temperature.csv": [
        ("field.temperature", "Temperatura medida pelo sensor de IMU (°C)."),
        ("field.variance", "Variância/Incerteza associada à medida de temperatura."),
    ],
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-local_position-odom.csv": [
        ("field.child_frame_id", "Frame do corpo do veículo (normalmente 'base_link')."),
        ("field.pose.pose.position.x/y/z", "Posição estimada no frame local (metros)."),
        ("field.pose.pose.orientation.x/y/z/w", "Orientação em Quaternions. Evita o 'Gimbal Lock' e representa a atitude do drone."),
        ("field.pose.covariance", "Matriz 6x6 (achatada) de incerteza da Pose (posição + orientação)."),
        ("field.twist.twist.linear.x/y/z", "Velocidade linear nos eixos X, Y e Z (m/s)."),
        ("field.twist.twist.angular.x/y/z", "Velocidade angular (Taxa de giro) em rad/s."),
        ("field.twist.covariance", "Matriz 6x6 de incerteza do Twist (velocidades)."),
    ],
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-airspeed.csv": [
        ("field.commanded", "Velocidade do ar (Airspeed) desejada pelo controlador (m/s)."),
        ("field.measured", "Velocidade do ar efetivamente medida pelo sensor pitot (m/s)."),
    ],
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-errors.csv": [
        ("field.alt_error", "Diferença entre altitude desejada e estimada (metros)."),
        ("field.aspd_error", "Diferença entre airspeed desejada e medida (m/s)."),
        ("field.xtrack_error", "Erro lateral (Cross-track) em relação à linha da rota (metros)."),
        ("field.wp_dist", "Distância linear restante até o próximo Waypoint (metros)."),
    ],
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-pitch.csv": [
        ("field.commanded", "Ângulo de Arfagem (Pitch) desejado pelo controle (radianos)."),
        ("field.measured", "Ângulo de Arfagem (Pitch) real estimado (radianos)."),
    ],
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-roll.csv": [
        ("field.commanded", "Ângulo de Rolagem (Roll) desejado pelo controle (radianos)."),
        ("field.measured", "Ângulo de Rolagem (Roll) real estimado (radianos)."),
    ],
}

def write_md(path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Documentação do Dataset ALFA / MAVROS",
        f"\n*Gerado automaticamente em: {now}*",
        "\nEste guia descreve o significado das colunas nos arquivos de telemetria.",
        "\n---",
        "## 1. Colunas Comuns (Metadados)",
        "Estas colunas estão presentes em quase todos os arquivos e seguem o padrão de mensagens do ROS:",
        "\n| Coluna | Descrição |",
        "| :--- | :--- |"
    ]
    
    for col, desc in COMMON_COLUMNS:
        lines.append(f"| `{col}` | {desc} |")

    lines.append("\n---")
    lines.append("## 2. Descrição por Arquivo Específico")

    for fname, items in FILES.items():
        lines.append(f"\n### {fname}")
        lines.append("| Coluna | Descrição |")
        lines.append("| :--- | :--- |")
        for col, desc in items:
            lines.append(f"| `{col}` | {desc} |")
    
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    out_path = OUT_DIR / "alfa_mavros_columns_explained.md"
    write_md(out_path)
    print(f"✅ Sucesso! Documentação didática gerada em: {out_path}")

if __name__ == "__main__":
    main()