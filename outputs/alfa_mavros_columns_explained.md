# Documentação do Dataset ALFA / MAVROS

*Gerado automaticamente em: 2026-02-11 02:50*

Este guia descreve o significado das colunas nos arquivos de telemetria.

---
## 1. Colunas Comuns (Metadados)
Estas colunas estão presentes em quase todos os arquivos e seguem o padrão de mensagens do ROS:

| Coluna | Descrição |
| :--- | :--- |
| `%time` | Timestamp do registro no CSV (nanossegundos). Instante da gravação no arquivo. |
| `field.header.seq` | Número sequencial da mensagem ROS. Usado para detectar perda de pacotes. |
| `field.header.stamp` | Timestamp do ROS (segundos/nanossegundos). Quando a mensagem foi gerada no sistema. |
| `field.header.frame_id` | Frame de referência (ex: base_link, map, local_origin). |

---
## 2. Descrição por Arquivo Específico

### carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-imu-temperature.csv
| Coluna | Descrição |
| :--- | :--- |
| `field.temperature` | Temperatura medida pelo sensor de IMU (°C). |
| `field.variance` | Variância/Incerteza associada à medida de temperatura. |

### carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-local_position-odom.csv
| Coluna | Descrição |
| :--- | :--- |
| `field.child_frame_id` | Frame do corpo do veículo (normalmente 'base_link'). |
| `field.pose.pose.position.x/y/z` | Posição estimada no frame local (metros). |
| `field.pose.pose.orientation.x/y/z/w` | Orientação em Quaternions. Evita o 'Gimbal Lock' e representa a atitude do drone. |
| `field.pose.covariance` | Matriz 6x6 (achatada) de incerteza da Pose (posição + orientação). |
| `field.twist.twist.linear.x/y/z` | Velocidade linear nos eixos X, Y e Z (m/s). |
| `field.twist.twist.angular.x/y/z` | Velocidade angular (Taxa de giro) em rad/s. |
| `field.twist.covariance` | Matriz 6x6 de incerteza do Twist (velocidades). |

### carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-airspeed.csv
| Coluna | Descrição |
| :--- | :--- |
| `field.commanded` | Velocidade do ar (Airspeed) desejada pelo controlador (m/s). |
| `field.measured` | Velocidade do ar efetivamente medida pelo sensor pitot (m/s). |

### carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-errors.csv
| Coluna | Descrição |
| :--- | :--- |
| `field.alt_error` | Diferença entre altitude desejada e estimada (metros). |
| `field.aspd_error` | Diferença entre airspeed desejada e medida (m/s). |
| `field.xtrack_error` | Erro lateral (Cross-track) em relação à linha da rota (metros). |
| `field.wp_dist` | Distância linear restante até o próximo Waypoint (metros). |

### carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-pitch.csv
| Coluna | Descrição |
| :--- | :--- |
| `field.commanded` | Ângulo de Arfagem (Pitch) desejado pelo controle (radianos). |
| `field.measured` | Ângulo de Arfagem (Pitch) real estimado (radianos). |

### carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-roll.csv
| Coluna | Descrição |
| :--- | :--- |
| `field.commanded` | Ângulo de Rolagem (Roll) desejado pelo controle (radianos). |
| `field.measured` | Ângulo de Rolagem (Roll) real estimado (radianos). |