import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SPLIT_DATE = "2023-01-01"

FEATURES_WITH_TAX = [
    "brent_lag1",
    "brent_lag2",
    "fx_lag1",
    "tax_lag1",
    "ipc_lag1",
    "pvp_lag1",
    "pvp_lag2",
]

FEATURES_WITHOUT_TAX = [
    "brent_lag1",
    "brent_lag2",
    "fx_lag1",
    "ipc_lag1",
    "pvp_lag1",
    "pvp_lag2",
]

MODEL_SPECS = [
    ("OLS", LinearRegression()),
    (
        "RandomForest",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
        ),
    ),
    (
        "GradientBoosting",
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=42,
        ),
    ),
]


def preparar_dades():
    df = pd.read_csv("dataset_mensual_final.csv")
    df["date"] = pd.to_datetime(df["month"].str.replace(":", "-") + "-01")
    df = df.sort_values("date").reset_index(drop=True)

    # Només deixem informació disponible fins al mes anterior.
    df["pvp_lag1"] = df["pvp"].shift(1)
    df["pvp_lag2"] = df["pvp"].shift(2)
    df["brent_lag1"] = df["brent"].shift(1)
    df["brent_lag2"] = df["brent"].shift(2)
    df["fx_lag1"] = df["fx"].shift(1)
    df["tax_lag1"] = df["tax"].shift(1)
    df["ipc_lag1"] = df["ipc_subyacente"].shift(1)

    return df.dropna().reset_index(drop=True)


def construir_dades_seguent_mes(last_row):
    return pd.DataFrame([{
        "brent_lag1": last_row["brent"],
        "brent_lag2": last_row["brent_lag1"],
        "fx_lag1": last_row["fx"],
        "tax_lag1": last_row["tax"],
        "ipc_lag1": last_row["ipc_subyacente"],
        "pvp_lag1": last_row["pvp"],
        "pvp_lag2": last_row["pvp_lag1"],
    }])


def obtenir_importancia(model, features):
    if hasattr(model, "coef_"):
        return pd.DataFrame({
            "feature": features,
            "weight": model.coef_,
        }).sort_values("weight", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature": features,
            "weight": model.feature_importances_,
        }).sort_values("weight", ascending=False).reset_index(drop=True)

    return pd.DataFrame({"feature": features})


def entrenar_i_avaluar(model_name, estimator, train, test, features, label):
    model = clone(estimator)
    model.fit(train[features], train["pvp"])

    y_pred = model.predict(test[features])
    mae = mean_absolute_error(test["pvp"], y_pred)
    mse = mean_squared_error(test["pvp"], y_pred)
    r2 = r2_score(test["pvp"], y_pred)

    results = test[["date", "pvp"]].copy()
    results["prediction"] = y_pred
    results["error"] = results["pvp"] - results["prediction"]

    print(f"\n===== {model_name} | {label} =====")
    print("Observacions train:", len(train))
    print("Observacions test:", len(test))
    print("Features:", features)
    print(results)
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)
    print(obtenir_importancia(model, features))

    return {
        "model_name": model_name,
        "label": label,
        "features": features,
        "model": model,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "results": results,
    }


def predir_seguent_mes(model, next_month_data, features):
    return model.predict(next_month_data[features])[0]


def imprimir_resum_prediccio(run, next_month_data, next_month):
    features = run["features"]
    model = run["model"]
    detall = next_month_data[features].T.reset_index()
    detall.columns = ["feature", "value"]

    print(f"\n----- Resum prediccio {run['model_name']} | {run['label']} ({next_month}) -----")
    print(detall)

    if hasattr(model, "coef_"):
        print("Nota: aquest model és lineal; els weights indiquen l'efecte marginal estimat.")
    else:
        print("Nota: aquest model no és lineal; els weights són importàncies, no coeficients.")

    print("Prediccio final:", predir_seguent_mes(model, next_month_data, features))


def main():
    df = preparar_dades()
    train = df[df["date"] < SPLIT_DATE]
    test = df[df["date"] >= SPLIT_DATE]

    runs = []
    for model_name, estimator in MODEL_SPECS:
        runs.append(
            entrenar_i_avaluar(
                model_name=model_name,
                estimator=estimator,
                train=train,
                test=test,
                features=FEATURES_WITH_TAX,
                label="AMB TAX_LAG1",
            )
        )
        runs.append(
            entrenar_i_avaluar(
                model_name=model_name,
                estimator=estimator,
                train=train,
                test=test,
                features=FEATURES_WITHOUT_TAX,
                label="SENSE TAX_LAG1",
            )
        )

    last_row = df.iloc[-1]
    next_month = (last_row["date"] + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    next_month_data = construir_dades_seguent_mes(last_row)

    print(f"\n===== DADES D'ENTRADA PER PREDIR {next_month} =====")
    print(next_month_data)

    resum = []
    for run in runs:
        pred = predir_seguent_mes(run["model"], next_month_data, run["features"])
        run["next_prediction"] = pred
        imprimir_resum_prediccio(run, next_month_data, next_month)
        resum.append({
            "model": run["model_name"],
            "variant": run["label"],
            "R2_test": run["r2"],
            "MAE_test": run["mae"],
            "MSE_test": run["mse"],
            "pred_next_month": pred,
        })

    resum_df = pd.DataFrame(resum).sort_values(
        ["R2_test", "MAE_test"],
        ascending=[False, True],
    ).reset_index(drop=True)

    print("\n===== RESUM FINAL DE MODELS =====")
    print(resum_df)


if __name__ == "__main__":
    main()
