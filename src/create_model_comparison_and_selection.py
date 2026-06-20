from __future__ import annotations

import math
import re
from html import escape
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency for external users
    XGBRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_mensual_final.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLIT_DATE = "2023-01-01"
FINAL_FEATURES = [
    "brent_lag1",
    "fx",
    "covid_post",
    "brent_lag1_x_covid_post",
]


def euro_litre(value: float) -> str:
    return f"{value:.4f}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["month"] + "-01")
    df = df.sort_values("date").reset_index(drop=True)
    df["covid_post"] = (df["date"] >= "2020-03-01").astype(int)
    df["brent_lag1_x_covid_post"] = df["brent_lag1"] * df["covid_post"]
    return df.dropna(subset=["pvp", *FINAL_FEATURES]).reset_index(drop=True)


def fit_final_ols(df: pd.DataFrame):
    return sm.OLS(df["pvp"], sm.add_constant(df[FINAL_FEATURES])).fit(cov_type="HC3")


def coefficient_table(fit) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.values,
            "robust_se": fit.bse.values,
            "p_value": fit.pvalues.values,
            "ci_low": fit.conf_int()[0].values,
            "ci_high": fit.conf_int()[1].values,
        }
    )
    return out


def model_specs() -> list[tuple[str, str, object]]:
    specs: list[tuple[str, str, object]] = [
        ("OLS", "linear benchmark", LinearRegression()),
        (
            "Ridge",
            "regularized linear model",
            make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])),
        ),
        (
            "Lasso",
            "sparse regularized linear model",
            make_pipeline(
                StandardScaler(),
                LassoCV(alphas=[0.001, 0.01, 0.1, 1], cv=5, max_iter=10000),
            ),
        ),
        (
            "Elastic Net",
            "mixed regularized linear model",
            make_pipeline(
                StandardScaler(),
                ElasticNetCV(
                    alphas=[0.001, 0.01, 0.1, 1],
                    l1_ratio=[0.2, 0.5, 0.8],
                    cv=5,
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ),
        (
            "Decision Tree",
            "simple non-linear tree",
            DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=42),
        ),
        (
            "Random Forest",
            "bagged tree ensemble",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42,
            ),
        ),
        (
            "Gradient Boosting",
            "boosted tree ensemble",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=3,
                random_state=42,
            ),
        ),
        (
            "SVR",
            "kernel method",
            make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.02)),
        ),
    ]
    if XGBRegressor is not None:
        specs.insert(
            -1,
            (
                "XGBoost",
                "boosted tree ensemble",
                XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=2,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        )
    return specs


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2_holdout": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < SPLIT_DATE].copy()
    test = df[df["date"] >= SPLIT_DATE].copy()
    rows = []
    frames = []
    for name, role, estimator in model_specs():
        model = clone(estimator)
        model.fit(train[FINAL_FEATURES], train["pvp"])
        pred = model.predict(test[FINAL_FEATURES])
        rows.append(
            {
                "model": name,
                "method_type": role,
                "n_train": len(train),
                "n_test": len(test),
                **metrics(test["pvp"], pred),
            }
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": test["date"],
                    "actual_pvp": test["pvp"].values,
                    "prediction": pred,
                    "error": pred - test["pvp"].values,
                    "model": name,
                }
            )
        )
    comparison = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    selection_roles = {
        "OLS": "Selected primary model",
        "Lasso": "Close linear challenger",
        "Ridge": "Close linear challenger",
        "Elastic Net": "Close linear challenger",
        "Gradient Boosting": "Rejected: higher error",
        "SVR": "Rejected: higher error",
        "Random Forest": "Rejected: higher error",
        "XGBoost": "Rejected: higher error",
        "Decision Tree": "Rejected: higher error",
    }
    comparison["selection_role"] = comparison["model"].map(selection_roles)
    predictions = pd.concat(frames, ignore_index=True)
    return comparison, predictions


def save_figures(df: pd.DataFrame, fit, comparison: pd.DataFrame, predictions: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fitted = fit.fittedvalues

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(df["date"], df["pvp"], label="Actual price", linewidth=2)
    ax.plot(df["date"], fitted, label="Final OLS fitted value", linewidth=2)
    ax.set_title("Gasoline 95 price and final regression fit")
    ax.set_ylabel("EUR/litre")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_actual_vs_fitted.png", dpi=180)
    plt.close(fig)

    comp = comparison.sort_values("MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(comp["model"], comp["MAE"], color="#456990")
    ax.invert_yaxis()
    ax.set_title("Holdout MAE by model")
    ax.set_xlabel("MAE, EUR/litre")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_mae_full.png", dpi=180)
    plt.close(fig)

    old_interpretability_chart = FIGURES_DIR / "model_comparison_interpretability.png"
    if old_interpretability_chart.exists():
        old_interpretability_chart.unlink()

    selected = predictions[predictions["model"].isin(["OLS", "Lasso", "XGBoost"])].copy()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    actual = selected[["date", "actual_pvp"]].drop_duplicates()
    ax.plot(actual["date"], actual["actual_pvp"], label="Actual", linewidth=2.5, color="#222222")
    for model_name in selected["model"].unique():
        frame = selected[selected["model"] == model_name]
        ax.plot(frame["date"], frame["prediction"], label=model_name, linewidth=1.8)
    ax.set_title("Holdout predictions")
    ax.set_ylabel("EUR/litre")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "holdout_predictions_selected.png", dpi=180)
    plt.close(fig)


def format_table_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.{float_digits}f}")
    display = display.fillna("").astype(str)
    widths = {
        col: max(len(col), *(len(value) for value in display[col].tolist()))
        for col in display.columns
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in display.columns) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in display.columns) + " |"
    rows = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_markdown_report(df: pd.DataFrame, fit, coefs: pd.DataFrame, comparison: pd.DataFrame) -> str:
    best = comparison.iloc[0]
    ols = comparison[comparison["model"] == "OLS"].iloc[0]
    mae_gap = ols["MAE"] - best["MAE"]
    brent_base = coefs.loc[coefs["term"] == "brent_lag1", "coef"].iloc[0]
    brent_post = coefs.loc[coefs["term"] == "brent_lag1_x_covid_post", "coef"].iloc[0]
    slope_after_covid = brent_base + brent_post

    coef_md = format_table_markdown(
        coefs.assign(
            coef=coefs["coef"].round(4),
            robust_se=coefs["robust_se"].round(4),
            p_value=coefs["p_value"].round(4),
            ci_low=coefs["ci_low"].round(4),
            ci_high=coefs["ci_high"].round(4),
        )
    )
    comp_cols = ["model", "method_type", "MAE", "RMSE", "R2_holdout", "Bias", "selection_role"]
    comp_md = format_table_markdown(comparison[comp_cols])

    text = dedent(
        f"""
        # Model Selection Memo

        ## Challenger review for the Gasoline 95 regression

        ### Decision

        Retain the final OLS regression with HC3 robust standard errors as the primary model.

        Lasso gives the lowest holdout MAE at **{best['MAE']:.4f} EUR/litre**. OLS gives **{ols['MAE']:.4f} EUR/litre**, only **{mae_gap:.4f} EUR/litre** higher. That gap is not material enough to replace the regression in a project where the model has to be explained, challenged and documented, not only ranked by a single predictive metric.

        The selected model also gives a clear economic reading. Before the COVID/post-COVID period, a one-dollar increase in lagged Brent is associated with an increase of about **{brent_base:.4f} EUR/litre** in the retail price. With the interaction term, the implied Brent slope during the COVID/post-COVID regime rises to about **{slope_after_covid:.4f} EUR/litre**. This is the main reason to keep the interaction explicit instead of hiding the relationship inside a more flexible estimator.

        ### Modelling Question

        The objective is to explain monthly Gasoline 95 prices in Lleida using a small set of economically meaningful variables, and then check whether common machine-learning challengers improve the out-of-sample result enough to justify a different model choice.

        The dataset contains monthly observations from **2016-01 to 2025-12**. After creating lagged variables, the effective modelling sample starts in **{df['month'].iloc[0]}** and contains **{len(df)} observations**.

        The dependent variable is `pvp`, the retail price of Gasoline 95 in Lleida, measured in EUR/litre. The final model uses:

        - `brent_lag1`: Brent crude oil price lagged one month.
        - `fx`: USD/EUR exchange rate.
        - `covid_post`: dummy equal to 1 from March 2020 onward.
        - `brent_lag1_x_covid_post`: interaction between lagged Brent and the COVID/post-COVID dummy.

        The tax variable was excluded from the final model. It improves in-sample fit, but it is too close to the retail price by construction and would make the model less useful as an independent explanation.

        ### Validation Design

        The challenger review keeps the information set fixed. OLS, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost and SVR all use exactly the same four final features. This isolates the estimator choice from feature-set changes.

        The split is chronological rather than random: observations before **2023-01-01** are used for training and observations from **2023-01-01** onward are used as holdout data. The main metrics are MAE and RMSE in EUR/litre, with holdout R-squared and average prediction bias used as secondary checks.

        ### Final Regression Specification

        ```text
        pvp_t = beta0
              + beta1 * brent_lag1_t
              + beta2 * fx_t
              + beta3 * covid_post_t
              + beta4 * brent_lag1_t * covid_post_t
              + error_t
        ```

        Standard errors are HC3 robust standard errors.

        {coef_md}

        The regression has an in-sample R-squared of **{fit.rsquared:.4f}** and an adjusted R-squared of **{fit.rsquared_adj:.4f}**. The result should not be read as a causal estimate. It is an explanatory model built from a small monthly time series and intended to summarize the main price relationships in an interpretable way.

        ![Actual price and final OLS fitted value](../figures/regression_actual_vs_fitted.png)

        ### Challenger Results

        The strongest challengers are the regularized linear models. They are useful checks, but they do not change the modelling decision because their improvement over OLS is very small.

        {comp_md}

        ![Holdout MAE by model](../figures/model_comparison_mae_full.png)

        Tree-based and kernel-based models do not improve the result on this feature set. XGBoost is included as a standard high-capacity benchmark, but it performs worse than the linear alternatives in this sample. That result is consistent with the dataset size and with the fact that the selected features already describe a mostly linear economic relationship.

        ![Holdout predictions for selected models](../figures/holdout_predictions_selected.png)

        ### Selection Rationale

        OLS is retained because it gives the best balance between evidence and usability:

        - The out-of-sample error is almost identical to the best regularized challenger.
        - The coefficients remain directly explainable in economic terms.
        - The Brent interaction makes the regime change visible and auditable.
        - The model is simpler to document and challenge than the non-linear alternatives.
        - The more flexible models do not provide enough empirical benefit on this dataset.

        Ridge, Lasso and Elastic Net are credible challengers. For a pure forecasting task, one of them could be selected after further validation. For this project, the marginal error reduction is not enough to offset the loss of direct coefficient interpretation.

        ### Limitations

        The holdout set has only 36 monthly observations, so the exact ranking of close models should not be overinterpreted. The sample also contains unusual market shocks, especially COVID and the 2022 energy-price period. The model is explanatory, not causal, and it should not be treated as a production forecasting system.

        A stronger validation exercise would add rolling-origin validation, alternative lag structures, logarithmic transformations and richer time-series specifications. Those extensions are outside the scope of this project but are the natural next tests before any operational use.

        ### Conclusion

        The final regression is selected because it is accurate enough, transparent, economically coherent and easier to challenge. The model comparison does not show a clear enough benefit from more complex methods to replace it.
        """
    ).strip()
    text = "\n".join(line[8:] if line.startswith("        ") else line for line in text.splitlines())
    return text + "\n"


def markdown_to_html(markdown: str) -> str:
    try:
        import markdown as md

        body = md.markdown(markdown, extensions=["tables", "fenced_code"])
    except Exception:
        body = "<pre>" + markdown.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Model Selection Memo</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 920px; margin: 40px auto; line-height: 1.55; color: #222; }}
    h1, h2, h3 {{ color: #111; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 18px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 7px 9px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code, pre {{ background: #f6f6f6; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    img {{ max-width: 100%; margin: 12px 0 22px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    safe = escape(text)
    safe = safe.replace("&lt;br/&gt;", "<br/>")
    safe = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", safe)
    return Paragraph(safe, style)


def pdf_table_from_df(df: pd.DataFrame, max_rows: int | None = None) -> Table:
    display = df.copy()
    if max_rows:
        display = display.head(max_rows)
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.4f}")
    values = [list(display.columns)] + display.astype(str).values.tolist()
    table = Table(values, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ]
        )
    )
    return table


def write_pdf_report(markdown_report: str, coefs: pd.DataFrame, comparison: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "model_comparison_and_selection.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="BodyJust", parent=styles["BodyText"], fontSize=9.5, leading=13))

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
    )

    story = [
        Paragraph("Model Selection Memo", styles["Title"]),
        Paragraph("Challenger review for the Gasoline 95 regression", styles["Heading2"]),
        Spacer(1, 8),
    ]

    sections = markdown_report.split("\n### ")
    intro = sections[0].split("\n", 3)[-1]
    story.append(paragraph(intro.replace("\n\n", "<br/><br/>"), styles["BodyJust"]))

    for section in sections[1:]:
        title, _, body = section.partition("\n")
        story.append(Spacer(1, 8))
        story.append(Paragraph(title, styles["Heading2"]))

        if title.startswith("Final Regression Specification"):
            story.append(paragraph("The final model uses HC3 robust standard errors.", styles["BodyJust"]))
            story.append(pdf_table_from_df(coefs[["term", "coef", "robust_se", "p_value"]]))
            story.append(Spacer(1, 8))
            story.append(Image(str(FIGURES_DIR / "regression_actual_vs_fitted.png"), width=16.2 * cm, height=8.1 * cm))
        elif title.startswith("Challenger Results"):
            story.append(
                paragraph(
                    "All challenger models use the same final features and the same chronological holdout split.",
                    styles["BodyJust"],
                )
            )
            story.append(
                pdf_table_from_df(
                    comparison[
                        [
                            "model",
                            "method_type",
                            "MAE",
                            "RMSE",
                            "R2_holdout",
                            "selection_role",
                        ]
                    ]
                )
            )
            story.append(Spacer(1, 8))
            story.append(Image(str(FIGURES_DIR / "model_comparison_mae_full.png"), width=16.2 * cm, height=8.1 * cm))
            story.append(Spacer(1, 8))
            story.append(Image(str(FIGURES_DIR / "holdout_predictions_selected.png"), width=16.2 * cm, height=8.1 * cm))
        else:
            cleaned = []
            for line in body.splitlines():
                if line.startswith("|") or line.startswith("![") or line.startswith("```"):
                    continue
                if line.strip():
                    cleaned.append(line.strip().lstrip("- "))
            text = "<br/><br/>".join(cleaned[:10])
            if text:
                story.append(paragraph(text, styles["BodyJust"]))

    doc.build(story)


def notebook_source(comparison: pd.DataFrame) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    cells = [
        nbf.v4.new_markdown_cell(
            "# Model Comparison Notebook\n\n"
            "This notebook supports the model-selection memo. The comparison is intentionally narrow: "
            "all challengers use the same final feature set, so the test is whether a different estimator improves the selected regression enough to justify changing the model."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import math\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import statsmodels.api as sm\n"
            "from sklearn.base import clone\n"
            "from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor\n"
            "from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
            "from sklearn.pipeline import make_pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.svm import SVR\n"
            "from sklearn.tree import DecisionTreeRegressor\n"
            "from xgboost import XGBRegressor\n\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'dataset_mensual_final.csv'\n"
            "SPLIT_DATE = '2023-01-01'\n"
            "FINAL_FEATURES = ['brent_lag1', 'fx', 'covid_post', 'brent_lag1_x_covid_post']"
        ),
        nbf.v4.new_code_cell(
            "df = pd.read_csv(DATA_PATH)\n"
            "df['date'] = pd.to_datetime(df['month'] + '-01')\n"
            "df = df.sort_values('date').reset_index(drop=True)\n"
            "df['covid_post'] = (df['date'] >= '2020-03-01').astype(int)\n"
            "df['brent_lag1_x_covid_post'] = df['brent_lag1'] * df['covid_post']\n"
            "df_model = df.dropna(subset=['pvp', *FINAL_FEATURES]).reset_index(drop=True)\n"
            "print(df_model[['month','pvp', *FINAL_FEATURES]].head())\n"
            "print('\\nEffective sample:', df_model['month'].iloc[0], 'to', df_model['month'].iloc[-1], '| n =', len(df_model))"
        ),
        nbf.v4.new_markdown_cell(
            "## Final Regression\n\n"
            "The final regression keeps the features selected in the econometric analysis. "
            "HC3 robust standard errors are used because the residual diagnostics suggested imperfect classical assumptions."
        ),
        nbf.v4.new_code_cell(
            "X = sm.add_constant(df_model[FINAL_FEATURES])\n"
            "ols_fit = sm.OLS(df_model['pvp'], X).fit(cov_type='HC3')\n"
            "print(ols_fit.summary())"
        ),
        nbf.v4.new_markdown_cell(
            "## Challenger Models\n\n"
            "The models below are compared on a chronological holdout split. The purpose is not aggressive tuning; "
            "it is to check whether common alternatives clearly beat the final regression using the same information set."
        ),
        nbf.v4.new_code_cell(
            "train = df_model[df_model['date'] < SPLIT_DATE].copy()\n"
            "test = df_model[df_model['date'] >= SPLIT_DATE].copy()\n\n"
            "models = [\n"
            "    ('OLS', 'linear benchmark', LinearRegression()),\n"
            "    ('Ridge', 'regularized linear model', make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01,0.1,1,10,100]))),\n"
            "    ('Lasso', 'sparse regularized linear model', make_pipeline(StandardScaler(), LassoCV(alphas=[0.001,0.01,0.1,1], cv=5, max_iter=10000))),\n"
            "    ('Elastic Net', 'mixed regularized linear model', make_pipeline(StandardScaler(), ElasticNetCV(alphas=[0.001,0.01,0.1,1], l1_ratio=[0.2,0.5,0.8], cv=5, max_iter=10000, random_state=42))),\n"
            "    ('Decision Tree', 'simple non-linear tree', DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=42)),\n"
            "    ('Random Forest', 'bagged tree ensemble', RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3, random_state=42)),\n"
            "    ('Gradient Boosting', 'boosted tree ensemble', GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, min_samples_leaf=3, random_state=42)),\n"
            "    ('XGBoost', 'boosted tree ensemble', XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.9, objective='reg:squarederror', random_state=42, n_jobs=1)),\n"
            "    ('SVR', 'kernel method', make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.02))),\n"
            "]\n\n"
            "selection_roles = {\n"
            "    'OLS': 'Selected primary model',\n"
            "    'Lasso': 'Close linear challenger',\n"
            "    'Ridge': 'Close linear challenger',\n"
            "    'Elastic Net': 'Close linear challenger',\n"
            "    'Gradient Boosting': 'Rejected: higher error',\n"
            "    'SVR': 'Rejected: higher error',\n"
            "    'Random Forest': 'Rejected: higher error',\n"
            "    'XGBoost': 'Rejected: higher error',\n"
            "    'Decision Tree': 'Rejected: higher error',\n"
            "}\n\n"
            "rows = []\n"
            "predictions = []\n"
            "for name, method_type, estimator in models:\n"
            "    model = clone(estimator)\n"
            "    model.fit(train[FINAL_FEATURES], train['pvp'])\n"
            "    pred = model.predict(test[FINAL_FEATURES])\n"
            "    rows.append({\n"
            "        'model': name,\n"
            "        'method_type': method_type,\n"
            "        'MAE': mean_absolute_error(test['pvp'], pred),\n"
            "        'RMSE': math.sqrt(mean_squared_error(test['pvp'], pred)),\n"
            "        'R2_holdout': r2_score(test['pvp'], pred),\n"
            "        'Bias': float(np.mean(pred - test['pvp'])),\n"
            "        'selection_role': selection_roles[name],\n"
            "    })\n"
            "    predictions.append(pd.DataFrame({'date': test['date'], 'actual': test['pvp'].values, 'prediction': pred, 'model': name}))\n\n"
            "comparison = pd.DataFrame(rows).sort_values(['MAE','RMSE']).reset_index(drop=True)\n"
            "predictions = pd.concat(predictions, ignore_index=True)\n"
            "comparison"
        ),
        nbf.v4.new_markdown_cell(
            "## Current Results\n\n"
            "The table below is generated by `src/create_model_comparison_and_selection.py` from the current dataset. "
            "It is included here so the notebook remains readable on GitHub even before re-running the cells.\n\n"
            + format_table_markdown(
                comparison[
                    [
                        "model",
                        "method_type",
                        "MAE",
                        "RMSE",
                        "R2_holdout",
                        "Bias",
                        "selection_role",
                    ]
                ]
            )
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "plot_df = comparison.sort_values('MAE')\n"
            "ax.barh(plot_df['model'], plot_df['MAE'])\n"
            "ax.invert_yaxis()\n"
            "ax.set_title('Holdout MAE by model')\n"
            "ax.set_xlabel('MAE, EUR/litre')\n"
            "ax.grid(axis='x', alpha=0.25)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "selected = predictions[predictions['model'].isin(['OLS', 'Lasso', 'XGBoost'])]\n"
            "actual = selected[['date', 'actual']].drop_duplicates()\n"
            "ax.plot(actual['date'], actual['actual'], label='Actual', linewidth=2.5, color='black')\n"
            "for model_name in selected['model'].unique():\n"
            "    frame = selected[selected['model'] == model_name]\n"
            "    ax.plot(frame['date'], frame['prediction'], label=model_name)\n"
            "ax.set_title('Holdout predictions: selected models')\n"
            "ax.set_ylabel('EUR/litre')\n"
            "ax.legend()\n"
            "ax.grid(alpha=0.25)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Conclusion\n\n"
            "Lasso and Ridge produce slightly lower holdout errors than OLS, but the improvement is very small. "
            "The selected model is therefore the final OLS regression: it remains close to the best predictive alternatives while keeping the coefficients and the COVID-Brent interaction directly explainable. "
            "The non-linear models, including XGBoost, do not justify their additional complexity on this small monthly dataset."
        ),
    ]
    nb.cells = cells
    return nb


def write_notebook(comparison: pd.DataFrame) -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS_DIR / "model_comparison.ipynb"
    nb = notebook_source(comparison)
    nbf.write(nb, path)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    fit = fit_final_ols(df)
    coefs = coefficient_table(fit)
    comparison, predictions = compare_models(df)

    coefs.to_csv(PROCESSED_DIR / "final_regression_coefficients.csv", index=False)
    comparison.to_csv(PROCESSED_DIR / "model_comparison_full.csv", index=False)
    predictions.to_csv(PROCESSED_DIR / "model_comparison_predictions_full.csv", index=False)

    save_figures(df, fit, comparison, predictions)
    markdown_report = write_markdown_report(df, fit, coefs, comparison)
    (REPORTS_DIR / "model_comparison_and_selection.md").write_text(markdown_report, encoding="utf-8")
    (REPORTS_DIR / "model_comparison_and_selection.html").write_text(
        markdown_to_html(markdown_report), encoding="utf-8"
    )
    write_pdf_report(markdown_report, coefs, comparison)
    write_notebook(comparison)

    print("Created:")
    print(REPORTS_DIR / "model_comparison_and_selection.md")
    print(REPORTS_DIR / "model_comparison_and_selection.pdf")
    print(NOTEBOOKS_DIR / "model_comparison.ipynb")
    print(comparison[["model", "MAE", "RMSE", "R2_holdout", "selection_role"]].to_string(index=False))


if __name__ == "__main__":
    main()
