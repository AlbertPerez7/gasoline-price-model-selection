from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def monthly_first_observation(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out["month"] = out[date_col].dt.to_period("M")
    return out.sort_values(date_col).groupby("month", as_index=False).first()


def load_brent(path: Path) -> pd.DataFrame:
    brent = pd.read_excel(path, sheet_name="Data 1", header=2, usecols=[0, 1])
    brent.columns = ["date", "brent"]
    brent = monthly_first_observation(brent, "date")
    brent["brent"] = pd.to_numeric(brent["brent"], errors="coerce")
    return brent[["month", "brent"]]


def load_gasoline_prices(path: Path) -> pd.DataFrame:
    gasoline = pd.read_csv(path)
    required = ["date", "product", "pvp", "tax"]
    missing = [col for col in required if col not in gasoline.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    gasoline = monthly_first_observation(gasoline, "date")
    gasoline["pvp"] = pd.to_numeric(gasoline["pvp"], errors="coerce")
    gasoline["tax"] = pd.to_numeric(gasoline["tax"], errors="coerce")
    return gasoline[["month", "product", "pvp", "tax"]]


def load_core_cpi(path: Path) -> pd.DataFrame:
    cpi = pd.read_csv(path, sep=";", decimal=",")
    if "PERIODO" not in cpi.columns or "VALOR" not in cpi.columns:
        raise ValueError(f"{path.name} does not contain the expected INE columns.")
    cpi = cpi[["PERIODO", "VALOR"]].rename(
        columns={"PERIODO": "date", "VALOR": "ipc_subyacente"}
    )
    cpi["date"] = cpi["date"].astype(str).str.replace(
        r"^(\d{4})M(\d{2})$", r"\1-\2-01", regex=True
    )
    cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
    cpi["month"] = cpi["date"].dt.to_period("M")
    cpi["ipc_subyacente"] = pd.to_numeric(cpi["ipc_subyacente"], errors="coerce")
    return cpi[["month", "ipc_subyacente"]].dropna()


def load_fx(path: Path) -> pd.DataFrame:
    fx = pd.read_csv(path)
    value_cols = [col for col in fx.columns if "US dollar/Euro" in col]
    if "DATE" not in fx.columns or not value_cols:
        raise ValueError(f"{path.name} does not contain the expected ECB columns.")
    fx = fx[["DATE", value_cols[0]]].rename(columns={"DATE": "date", value_cols[0]: "fx"})
    fx = monthly_first_observation(fx, "date")
    fx["fx"] = pd.to_numeric(fx["fx"], errors="coerce")
    return fx[["month", "fx"]]


def load_consumption(path: Path) -> pd.DataFrame:
    consumption = pd.read_csv(path, usecols=["month", "consum"])
    consumption["month"] = pd.to_datetime(
        consumption["month"], format="%Y-%m", errors="coerce"
    ).dt.to_period("M")
    consumption["consum"] = pd.to_numeric(consumption["consum"], errors="coerce")
    return consumption[["month", "consum"]].dropna()


def build_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    brent = load_brent(raw_dir / "brent.xls")
    gasoline = load_gasoline_prices(raw_dir / "gasolina.csv")
    cpi = load_core_cpi(raw_dir / "ipc_subyacente.csv")
    fx = load_fx(raw_dir / "fx.csv")
    consumption = load_consumption(raw_dir / "consum_lleida_gasolina95.csv")

    dataset = (
        brent.merge(gasoline, on="month", how="inner")
        .merge(cpi, on="month", how="inner")
        .merge(fx, on="month", how="inner")
        .merge(consumption, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True)
    )

    dataset["brent_lag1"] = dataset["brent"].shift(1)
    dataset["consum_lag1"] = dataset["consum"].shift(1)
    return dataset[
        [
            "month",
            "pvp",
            "brent",
            "brent_lag1",
            "tax",
            "ipc_subyacente",
            "fx",
            "consum",
            "consum_lag1",
        ]
    ]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "dataset_mensual_final.csv"
    dataset = build_dataset()
    dataset.to_csv(output_path, index=False)

    first_month = dataset["month"].iloc[0]
    last_month = dataset["month"].iloc[-1]
    print(f"Dataset written to: {output_path}")
    print(f"Rows: {len(dataset)} | Period: {first_month} to {last_month}")
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
