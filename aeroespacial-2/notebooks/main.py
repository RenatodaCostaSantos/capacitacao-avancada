import matplotlib.pyplot as plt
from pathlib import Path
from training import train_and_evaluate_report
from failure_types import filter_by_failure_type, get_failure_distribution, extract_failure_type

ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR / "../data/processed"
OUTPUT_DIR = ROOT_DIR / "plots_individuais"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_FILE = ROOT_DIR / "training_report.txt"

def print_distribution(distribution, title, total_label="TOTAL"):
    print("=" * 60)
    print(title)
    print("=" * 60)
    for failure_type, count in sorted(distribution.items()):
        print(f"  {failure_type:20s}: {count:3d} voos")
    print(f"  {total_label:20s}: {sum(distribution.values()):3d} voos")
    print()


def main(failure_types=None, grid_search=True):
    flight_paths = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])

    flight_paths = [p for p in flight_paths if extract_failure_type(p.name) != 'no_ground_truth']
    
    distribution = get_failure_distribution(flight_paths)
    print_distribution(distribution, "DISTRIBUIÇÃO DE FALHAS NO DATASET")
    
    if failure_types is not None:
        flight_paths = filter_by_failure_type(flight_paths, failure_types)
        print(f"Filtrando para tipo(s): {failure_types}")
        print(f"Voos selecionados: {len(flight_paths)}")
        
        filtered_dist = get_failure_distribution(flight_paths)
        print_distribution(filtered_dist, "DISTRIBUIÇÃO APÓS FILTRO", "SELECIONADOS")
    
    print(f"Total de voos para treinamento: {len(flight_paths)}\n")

    trained_models, results = train_and_evaluate_report(
        flight_paths,
        grid_search=grid_search,
        balance_method='oversample',
        report_file=REPORT_FILE
    )

    print(f"Relatório salvo em: {REPORT_FILE}")


if __name__ == "__main__":
    import sys

    argv = [a for a in sys.argv[1:] if a != "--quick"]
    quick = len(argv) != len(sys.argv) - 1

    failure_types = None
    if len(argv) > 0:
        if argv[0] in ['engine', 'elevator', 'aileron', 'rudder', 'no_failure']:
            failure_types = argv[0]
        elif argv[0] == 'help':
            print("\nUso: python3 main.py [tipo_de_falha] [--quick]")
            print("\nTipos de falha disponíveis:")
            print("  engine    - Falhas de motor")
            print("  elevator  - Falhas de elevador")
            print("  aileron   - Falhas de aileron")
            print("  rudder    - Falhas de rudder")
            print("  no_failure - Voos normais (sem falha)")
            print("  --quick   - Sem grid search (só para teste rápido)")
            print("\nExemplos:")
            print("  python3 main.py engine     # Apenas falhas de motor + normais")
            print("  python3 main.py             # Todos os tipos misturados")
            print("  python3 main.py --quick     # Treino rápido, mesmo dataset completo")
            sys.exit(0)

    main(failure_types=failure_types, grid_search=not quick)