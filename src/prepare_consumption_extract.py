from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "consumption"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "consum_lleida_gasolina95.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "fecha": "month",
            "Fecha": "month",
            "provincia": "province",
            "Provincia": "province",
            "tipo_producto": "product",
            "Tipo Producto": "product",
            "consumo": "consumption",
            "Consumo Tm": "consumption",
        }
    )


def read_consumption_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df = normalize_columns(df)
    required = ["month", "province", "product", "consumption"]
    if not set(required).issubset(df.columns):
        return pd.DataFrame(columns=required)
    return df[required]


def prepare_dataset(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("*.csv")):
        frame = read_consumption_file(path)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No valid consumption CSV files found in {input_dir}")

    df = pd.concat(frames, ignore_index=True)
    product_normalized = (
        df["product"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True)
    )
    df = df[
        df["province"].astype(str).str.lower().eq("lleida")
        & product_normalized.eq("GASOLINA AUTO. S/PB 95 I.O.")
    ].copy()

    df = df.rename(
        columns={"province": "provincia", "product": "producte", "consumption": "consum"}
    )
    df["consum"] = pd.to_numeric(df["consum"], errors="coerce")
    return df[["month", "provincia", "producte", "consum"]].dropna().drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the clean Lleida Gasoline 95 consumption extract."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    clean = prepare_dataset(args.input_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.sort_values("month").to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"Rows: {len(clean):,}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
