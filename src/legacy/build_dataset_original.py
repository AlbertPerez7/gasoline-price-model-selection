
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
#   df_final.to_csv("dataset_final.csv", index=False)

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def brent_data(path):
    brent = pd.read_excel(
        path,
        sheet_name="Data 1",
        header=2,
        usecols=[0,1]
    )
    brent.columns = ["date", "brent"]
    brent["date"]= pd.to_datetime(brent["date"], format="%Y-%m-%d", errors="coerce")
    brent["month"] = brent["date"].dt.to_period("M")
    brent = (
        brent.sort_values("date")
        .groupby("month", as_index=False)
        .first()
    )
    return brent[["month", "brent"]]


def gasolina_data(path):
    gasolina = pd.read_csv(path,header=0, usecols=[0,1,2,3,4,5])
    gasolina["date"]= pd.to_datetime(gasolina["date"], format="%Y-%m-%d", errors="coerce")
    gasolina["month"] = gasolina["date"].dt.to_period("M")
    gasolina = (
        gasolina.sort_values("date")
        .groupby("month", as_index=False)
        .first()
    )
    return gasolina[["month", "product", "pvp", "tax"]]

# print(gasolina_data(BASE_DIR /"gasolina.csv").head(10))

def ipc_subjacent_data(path):

    ipc= pd.read_csv(path, header=0, usecols=[6,7], decimal=",", sep=";")
    ipc.columns = ["date", "ipc_subyacente"]
    ipc["date"] = ipc["date"].str.replace(r"^(\d{4})M(\d{2})$", r"\1-\2-01", regex=True)
    ipc["date"]= pd.to_datetime(ipc["date"], format="%Y-%m-%d", errors="coerce")
    ipc["month"] = ipc["date"].dt.to_period("M")
    return ipc[["month", "ipc_subyacente"]]

# print(ipc_subjacent_data(BASE_DIR /"ipc_subyacente.csv").head(10))

def fx_data(path):
    
    fx=pd.read_csv(path, header=0,usecols=[0,2],sep=",")
    fx.columns = ["date", "fx"]
    fx["date"]= pd.to_datetime(fx["date"], format="%Y-%m-%d", errors="coerce")
    fx["month"] = fx["date"].dt.to_period("M")
    fx = (
        fx.sort_values("date")
        .groupby("month", as_index=False)
        .first()
    )
    return fx[["month", "fx"]]
def consum_data(path):
    consum = pd.read_csv(
        path,
        sep=",",
        header=0,
        usecols=[0,3]
    )

    consum["month"] = pd.to_datetime(
        consum["month"],
        format="%Y-%m",
        errors="coerce"
    ).dt.to_period("M")

    return consum[["month", "consum"]]


def unir_datos(brent, gasolina, ipc_subyacente, fx,consum):
    
    df = (
        brent.merge(gasolina, on="month", how="inner")
        .merge(ipc_subyacente, on="month", how="inner")
        .merge(fx, on="month", how="inner")
        .merge(consum, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True)
    )
    return df
if __name__ == "__main__":
    df=(unir_datos(brent_data(BASE_DIR /"brent.xls"), gasolina_data(BASE_DIR /"gasolina.csv"), ipc_subjacent_data(BASE_DIR /"ipc_subyacente.csv"), fx_data(BASE_DIR /"fx.csv"), consum_data(BASE_DIR /"consum_lleida_gasolina95.csv")))
    df["brent_lag1"] = df["brent"].shift(1)
    df["consum_lag1"] = df["consum"].shift(1)
    df = df[["month", "pvp", "brent", "brent_lag1", "tax", "ipc_subyacente", "fx", "consum", "consum_lag1"]]
    print(df.head(15))
    df.to_csv(BASE_DIR / "dataset_mensual_final.csv", index=False)
