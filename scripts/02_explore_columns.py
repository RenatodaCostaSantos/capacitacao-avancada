from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path("data/alfa/unzipped")
OUT_DIR = Path("outputs")

TARGET_FILES = [
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-imu-temperature.csv",
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-local_position-odom.csv",
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-local_position-pose.csv",
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-local_position-velocity.csv",
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-airspeed.csv",
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-errors.csv",
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-pitch.csv",
    "carbonZ_2018-07-18-15-53-31_1_engine_failure-mavros-nav_info-roll.csv",
]


def locate(root: Path, filename: str) -> Path | None:
    hits = list(root.rglob(filename))
    if not hits:
        return None
    return hits[0]


def summarize_column(s: pd.Series) -> dict:
    n_rows = int(len(s))
    n_null = int(s.isna().sum())

    s_num = pd.to_numeric(s, errors="coerce")
    n_num = int(s_num.notna().sum())

    info = {
        "dtype": str(s.dtype),
        "n_rows": n_rows,
        "n_null": n_null,
        "null_frac": (n_null / n_rows) if n_rows else None,
        "n_numeric": n_num,
    }

    if n_num > 0:
        info.update(
            {
                "min": float(s_num.min()),
                "max": float(s_num.max()),
                "mean": float(s_num.mean()),
                "std": float(s_num.std(ddof=1)) if n_num > 1 else 0.0,
            }
        )
    else:
        info["n_unique"] = int(s.nunique(dropna=True))

    return info


def main() -> None:
    rows: list[dict] = []

    for filename in TARGET_FILES:
        fp = locate(ROOT, filename)

        if fp is None:
            rows.append(
                {"file": filename, "status": "NOT_FOUND", "column": None}
            )
            continue

        try:
            df = pd.read_csv(fp)
        except Exception as e:
            rows.append(
                {"file": fp.name, "status": f"READ_ERROR: {e}", "column": None}
            )
            continue

        for col in df.columns:
            base = {"file": fp.name, "status": "OK", "column": col}
            base.update(summarize_column(df[col]))
            rows.append(base)

    out = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = OUT_DIR / "alfa_selected_columns_summary.csv"
    out.to_csv(out_csv, index=False)

    ok = out[out["status"] == "OK"]

    overview = (
        ok.groupby("file")["column"]
        .apply(list)
        .reset_index()
        .rename(columns={"column": "columns"})
    )

    overview_csv = OUT_DIR / "alfa_selected_files_columns_list.csv"
    overview.to_csv(overview_csv, index=False)

    print("Arquivos gerados:")
    print(" -", out_csv)
    print(" -", overview_csv)

    missing = out[out["status"] == "NOT_FOUND"]["file"].dropna().unique().tolist()
    if missing:
        print("\nArquivos não encontrados:")
        for m in missing:
            print(" -", m)

    print("\nColunas por arquivo:")
    for _, r in overview.iterrows():
        print(f"\n{r['file']}")
        for c in r["columns"]:
            print("  -", c)


if __name__ == "__main__":
    main()
