# Model Decision Note

## Recommendation

Use the final OLS regression with HC3 robust standard errors as the analytical baseline.

The model should not be treated as a production forecasting model without further validation.

## Model Under Review

The final regression uses:

- lagged Brent crude oil price;
- USD/EUR exchange rate;
- COVID/post-COVID dummy;
- interaction between lagged Brent and the COVID/post-COVID dummy.

This specification comes from the econometric regression analysis, where the model was selected after considering multicollinearity, heteroskedasticity, variable relevance and interpretability.

## Challenger Review

All challenger models use the same final feature set as the selected regression. This keeps the review focused on estimator choice rather than feature engineering.

The validation split is chronological:

- training period: observations before 2023-01-01;
- holdout period: observations from 2023-01-01 onward.

The main metrics are MAE and RMSE in EUR/litre. Holdout R-squared and average prediction bias are secondary checks.

The challenger set includes Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost and Support Vector Regression.

## Result

Lasso gives the lowest holdout MAE, but the difference versus OLS is approximately 0.0006 EUR/litre. Ridge and Elastic Net are also close. The non-linear and kernel-based models do not improve the holdout result on this sample.

The gain from the regularized linear models is not large enough to replace the OLS specification. OLS remains easier to review, easier to explain and more directly connected to the economic interpretation of the Brent effect before and after the COVID/post-COVID regime.

## Main Model Risks

- The holdout set contains only 36 monthly observations.
- The sample includes unusual shocks, especially COVID and the 2022 energy-price period.
- The model is explanatory rather than causal.
- The challenger review uses the final regression feature set for all models, so it does not test every possible feature-engineering alternative.
- Further validation would be required before operational use.

## Decision

Retain the final OLS regression as the baseline model. Keep Ridge, Lasso and Elastic Net as documented challengers. Do not select the tree-based or kernel-based models on the current evidence.
