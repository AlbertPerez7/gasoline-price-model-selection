#!/usr/bin/env python3
"""
Uneix tots els CSV anuals de CNMC, selecciona les columnes útils i construeix la
base neta per al treball d'econometria.

Ús bàsic:
    python cnmc_prepare.py --input-dir /ruta/als/csvs --output-csv cnmc_lleida_95_2016_2026.csv

Si vols canviar província o producte:
    python cnmc_prepare.py --input-dir /ruta/als/csvs --province "Lleida" --product "Gasolina 95 E5"

El script:
1) llegeix tots els .csv de la carpeta
2) detecta l'estructura CNMC
3) filtra la província i el producte
4) renomena columnes
5) crea tax = pvp - pai
6) guarda un únic CSV final
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


EXPECTED_COLUMNS = {
    "Fecha Precio": "date",
    "Provincia": "province",
    "Producto": "product",
    "Promedio de Pai Diario CUBO €/litro": "pai",
    "Promedio de Pvp Diario CUBO €/litro": "pvp",
}


def normalize_text(s: str) -> str:
    """Normalitza text per comparar sense problemes de majúscules/minúscules."""
    return " ".join(str(s).strip().lower().split())


def read_cnmc_csv(path: Path) -> pd.DataFrame:
    """
    Llegeix un CSV de CNMC amb:
    - separador ';'
    - coma decimal ','
    """
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"El fitxer {path.name} no té les columnes esperades. Falten: {missing}\n"
            f"Columnes trobades: {list(df.columns)}"
        )

    df = df[list(EXPECTED_COLUMNS.keys())].rename(columns=EXPECTED_COLUMNS)
    df["source_file"] = path.name
    return df


def load_all_csvs(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No s'ha trobat cap CSV a: {input_dir}")

    frames = []
    for file in csv_files:
        try:
            frames.append(read_cnmc_csv(file))
        except Exception as exc:
            print(f"[ERROR] No s'ha pogut llegir {file.name}: {exc}", file=sys.stderr)
            raise

    combined = pd.concat(frames, ignore_index=True)
    return combined


def prepare_dataset(
    df: pd.DataFrame,
    province: str,
    product: str,
) -> pd.DataFrame:
    # Tipus
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pai"] = pd.to_numeric(df["pai"], errors="coerce")
    df["pvp"] = pd.to_numeric(df["pvp"], errors="coerce")

    # Eliminar files amb dades clau buides
    df = df.dropna(subset=["date", "province", "product", "pai", "pvp"]).copy()

    # Filtre flexible per província i producte
    province_norm = normalize_text(province)
    product_norm = normalize_text(product)

    df = df[
        df["province"].map(normalize_text).eq(province_norm)
        & df["product"].map(normalize_text).eq(product_norm)
    ].copy()

    if df.empty:
        raise ValueError(
            "Després de filtrar no queda cap fila.\n"
            f"Província cercada: {province}\n"
            f"Producte cercat: {product}\n"
            "Comprova els noms exactes dins dels CSV."
        )

    # Variable fiscal observada
    df["tax"] = df["pvp"] - df["pai"]

    # Ordenació i columnes finals
    df = df.sort_values("date").reset_index(drop=True)

    # Columnes finals netes
    final_cols = ["date", "province", "product", "pai", "pvp", "tax", "source_file"]
    df = df[final_cols]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uneix CSVs anuals de CNMC i construeix una base neta."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Carpeta on tens tots els CSVs descarregats de CNMC.",
    )
    parser.add_argument(
        "--output-csv",
        default="cnmc_lleida_95_clean.csv",
        type=Path,
        help="Nom/ruta del CSV de sortida.",
    )
    parser.add_argument(
        "--province",
        default="Lleida",
        help='Província a filtrar. Per defecte: "Lleida"',
    )
    parser.add_argument(
        "--product",
        default="Gasolina 95 E5",
        help='Producte a filtrar. Per defecte: "Gasolina 95 E5"',
    )

    args = parser.parse_args()

    combined = load_all_csvs(args.input_dir)
    clean = prepare_dataset(combined, province=args.province, product=args.product)

    # Guardar
    clean.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("\nBase creada correctament.")
    print(f"Files finals: {len(clean):,}")
    print(f"Rang de dates: {clean['date'].min().date()} -> {clean['date'].max().date()}")
    print(f"CSV guardat a: {args.output_csv.resolve()}")
    print("\nPrimeres files:")
    print(clean.head().to_string(index=False))


if __name__ == "__main__":
    main()
