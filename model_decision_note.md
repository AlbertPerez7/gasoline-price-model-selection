# Model Decision Note

## Decision

Retain the final OLS regression with HC3 robust standard errors as the primary model.

The model uses lagged Brent crude oil price, USD/EUR exchange rate, a COVID/post-COVID dummy, and the interaction between lagged Brent and that regime dummy.

## Validation Setup

All challenger models use the same final feature set as the selected regression. This keeps the comparison focused on the estimator choice rather than changing the information available to each model.

The validation split is chronological:

- training period: observations before 2023-01-01;
- holdout period: observations from 2023-01-01 onward.

The main metrics are MAE and RMSE in EUR/litre. Holdout R-squared and average prediction bias are used as secondary checks.

## Challenger Models

The challenger set includes Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost and Support Vector Regression.

Lasso gives the lowest holdout MAE, but the difference versus OLS is approximately 0.0006 EUR/litre. Ridge and Elastic Net are also close. The non-linear and kernel-based models do not improve the holdout result on this sample.

## Rationale

OLS is selected because the small predictive loss versus the best challenger is not large enough to justify moving to a less directly explainable model.

The selected regression is easier to audit, easier to document, and gives a direct interpretation of the Brent effect before and after the COVID/post-COVID regime. This matters because the purpose of the project is model explanation and selection, not only short-term forecast optimization.

## Main Limitations

- The holdout set contains only 36 monthly observations.
- The sample includes unusual shocks, especially COVID and the 2022 energy-price period.
- The model is explanatory rather than causal.
- The comparison uses the final regression feature set for all models, so it does not test every possible feature-engineering alternative.
- A stronger validation exercise would add rolling-origin validation and alternative lag or transformation choices.

## Final Assessment

The final OLS regression is the most defensible main model for this project. Regularized linear models are credible challengers, but they do not improve the holdout result enough to replace a transparent model whose assumptions and limitations can be explained.
