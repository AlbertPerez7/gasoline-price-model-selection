
# PIPELINE ETL PANDAS:

# 1. Definir rutes
#    BASE_DIR = Path(__file__).resolve().parent
#    path = BASE_DIR / "fitxer.csv"

# 2. Carregar dades a DataFrames
#    pd.read_excel(path, sheet_name=..., header=..., usecols=...)
#    pd.read_csv(path, header=..., usecols=..., sep=..., decimal=...)
#  Aquí ja es decideix:
#    - quina sheet llegir
#    - on està el header
#    - quines columnes vols
#    - quin separador/decimal té el CSV

# 3. Renombrar i estandarditzar columnes
#    df.columns = [...]  o   df = df.rename(columns={...})

#    Objectiu: tenir noms coherents com date, price, brent, fx, etc.

# 4. Convertir tipus i formats
#    pd.to_datetime(...)
#    pd.to_numeric(...)
#    .astype(...)
#    .str.replace(...)
#    Objectiu: dates com datetime, números com float/int, textos ben normalitzats.

# 5. Crear clau comuna o granularitat comuna
#    df["month"] = df["date"].dt.to_period("M")
#    df = df.sort_values("date").groupby("month", as_index=False).first()
#  Objectiu: que tots els datasets es puguin unir pel mateix camp.

# 6. Netejar i validar dades
#    df.dropna(subset=[...])      → elimina files amb valors buits (NaN)
#    df.fillna(...)               → omple valors buits (ex: ffill = valor anterior)
#    df.drop_duplicates()         → elimina files repetides

#    df.head()                    → veure primeres files
#    df.info()                    → veure tipus i buits
#    df.dtypes                    → veure tipus de dades

#    Objectiu:  eliminar errors, evitar dades inconsistents i comprovar que el DataFrame és correcte abans de continuar.

# 7. Unir i exportar
#    df_final = df1.merge(df2, on="month", how="inner")
#    df_final = df_final.merge(df3, on="month", how="inner")
#   df_final.to_csv("consum_lleida_gasolina95.csv", index=False)

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "consum_lleida_gasolina95.csv"

def consum_data(path):
    consum = pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        header=0,
    )

    consum = consum.rename(columns={
        "fecha": "month",
        "Fecha": "month",
        "provincia": "provincia",
        "Provincia": "provincia",
        "tipo_producto": "producte",
        "Tipo Producto": "producte",
        "consumo": "consum",
        "Consumo Tm": "consum",
    })

    columnes = ["month", "provincia", "producte", "consum"]
    if not set(columnes).issubset(consum.columns):
        print(f"Saltat {path.name}: no es un CSV de consum")
        return pd.DataFrame(columns=columnes)

    consum = consum[columnes]
    producte_normalitzat = consum["producte"].str.upper().str.replace(r"\s+", " ", regex=True)

    consum = consum[
        (consum["provincia"].str.lower() == "lleida") &
        (producte_normalitzat == "GASOLINA AUTO. S/PB 95 I.O.")
    ]

    return consum 

data_frames = []

for fitxer in BASE_DIR.glob("*.csv"):


    df = consum_data(fitxer)

    if not df.empty:
        data_frames.append(df)

if not data_frames:
    raise ValueError("No s'ha trobat cap CSV valid amb dades de consum.")

df_final = pd.concat(data_frames, ignore_index=True)
df_final = df_final.drop_duplicates()
df_final = df_final.sort_values("month")

print(df_final.head(120).to_string(index=False))
df_final.to_csv(OUTPUT_FILE, index=False)
