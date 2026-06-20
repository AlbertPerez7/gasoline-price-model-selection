from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "cnmc"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "gasolina.csv"

EXPECTED_COLUMNS = {
    "Fecha Precio": "date",
    "Provincia": "province",
    "Producto": "product",
    "Promedio de Pai Diario CUBO €/litro": "pai",
    "Promedio de Pvp Diario CUBO €/litro": "pvp",
}


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def read_cnmc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected CNMC columns: {missing}")
    return df[list(EXPECTED_COLUMNS)].rename(columns=EXPECTED_COLUMNS)


def prepare_dataset(input_dir: Path, province: str, product: str) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CNMC CSV files found in {input_dir}")

    frames = []
    for path in files:
        frame = read_cnmc_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pai"] = pd.to_numeric(df["pai"], errors="coerce")
    df["pvp"] = pd.to_numeric(df["pvp"], errors="coerce")
    df = df.dropna(subset=["date", "province", "product", "pai", "pvp"])

    df = df[
        df["province"].map(normalize_text).eq(normalize_text(province))
        & df["product"].map(normalize_text).eq(normalize_text(product))
    ].copy()

    if df.empty:
        raise ValueError(f"No rows found for province={province!r}, product={product!r}")

    df["tax"] = df["pvp"] - df["pai"]
    return df.sort_values("date")[
        ["date", "province", "product", "pai", "pvp", "tax", "source_file"]
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the clean Lleida Gasoline 95 extract from downloaded CNMC CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--province", default="Lleida")
    parser.add_argument("--product", default="Gasolina 95 E5")
    args = parser.parse_args()

    clean = prepare_dataset(args.input_dir, args.province, args.product)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"Rows: {len(clean):,}")
    print(f"Period: {clean['date'].min().date()} to {clean['date'].max().date()}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
