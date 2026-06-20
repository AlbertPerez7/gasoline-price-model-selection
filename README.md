# Gasoline 95 Price Regression and Model Review

Applied econometrics project on monthly Gasoline 95 retail prices in Lleida, Spain, from 2016 to 2025.

The repository contains the original regression report, a reproducible data pipeline, and a challenger-model review of the final linear regression.

The analysis first develops an interpretable linear regression using standard econometric diagnostics, including checks around multicollinearity, heteroskedasticity and variable relevance. A second step compares the selected regression against common statistical and machine-learning challengers using the same final feature set.

## Main Files

- `reports/regression_model_analysis.pdf`: English version of the original econometric regression report, preserving the original layout.
- `reports/model_comparison_and_selection.pdf`: model review note comparing the final regression against challenger models.
- `model_decision_note.md`: concise decision note covering the selected model, challenger results, validation setup and limitations.
- `notebooks/model_comparison.ipynb`: notebook comparing OLS with Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost and SVR.
- `src/build_dataset.py`: pandas pipeline that rebuilds the final monthly dataset from clean source extracts.
- `src/create_model_comparison_and_selection.py`: regenerates the model-comparison extension, figures, comparison tables and notebook.

## Main Finding

Lasso and Ridge obtain slightly lower holdout error than the final OLS regression, but the improvement is very small. The final OLS regression is retained as the analytical baseline because it is easier to interpret, easier to review, and almost as accurate as the best regularized alternatives.

XGBoost and other non-linear models are included as challengers, but they do not improve performance on this small monthly dataset when restricted to the same final features.

## Why OLS Is Retained

The final model is selected because it offers the best balance between:

- predictive performance;
- interpretability;
- robustness for a small monthly dataset;
- ease of review and explanation;
- economic coherence of the coefficients.

The project deliberately does not select the most complex model. The final choice is based on the modelling objective, the small holdout sample, the marginal gain from challengers, and the need for a transparent specification.

## Scope and Limitations

This is an applied econometrics and model-selection project, not a production fuel-price forecasting system. The holdout period contains only 36 monthly observations, so the model ranking should be interpreted cautiously. The final regression is explanatory rather than causal.

Future extensions could test rolling-origin validation, additional lag structures, logarithmic transformations, alternative structural-break definitions and richer time-series models.

## Reproduce

```bash
pip install -r requirements.txt
python src/build_dataset.py
python src/create_model_comparison_and_selection.py
```

The optional extraction scripts `src/prepare_cnmc_extract.py` and `src/prepare_consumption_extract.py` can be used to rebuild the clean raw extracts from the full public CSV downloads. Those full downloads are not included in the repository.

## Project Structure

- `data/raw/`: clean extracts from public data sources.
- `data/processed/`: final modelling dataset and generated comparison outputs.
- `figures/`: charts used in the report and notebook.
- `reports/`: original regression report and model-comparison extension.
- `notebooks/`: model-comparison notebook.
- `src/legacy/`: original scripts from the academic project.

## Data Sources

- CNMC: provincial daily fuel prices.
- U.S. Energy Information Administration: Brent crude oil prices.
- European Central Bank: USD/EUR exchange rate.
- INE: core CPI.
- datos.gob.es: monthly provincial petroleum consumption.
