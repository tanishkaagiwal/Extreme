# Extreme
Datathon 2025 : PS3 : Wealth Management Strategies


TASK 2 ( REGRESSION)

Objective:
To forecast each client's portfolio_value for the next 3 years using their 36-month historical time series and static attributes.

Approach:
EDA & Preprocessing:

Merged static_client_data and time_series_data.

Performed trend analysis, autocorrelation checks, and identified non-stationary patterns.

Observed that analyzing features globally caused temporal trends to cancel out — hence, processed data client-wise.

Stationarity & Transformation:

Applied differencing to achieve stationarity client-wise.

Engineered temporal features including:

Lag values.

Rolling statistics (mean, std over 3 and 6 months).

Feature Selection:

Used Extra Trees Regressor to select the most impactful features from both static and dynamic datasets.

Modeling:

Implemented an LSTM model trained on selected features.

 Initial Results:
R² score for forecasted_value_year_1: 0.4745

R² score for forecasted_value_year_2: 0.5614

R² score for forecasted_value_year_3: 0.4653

Feature Engineering for Final Model:
Introduced intelligent features to enhance predictive power:

portfolio_growth_rate:
df.groupby('client_id')['portfolio_value_stationary'].pct_change()

contribution_ratio:
monthly_contribution_stationary / (portfolio_value_stationary + 1e-6)

eq_fi_ratio:
equity_allocation_pct_stationary / (fixed_income_allocation_pct_stationary + 1e-6)

Final Dataset:
Used: stationary_data_with_smart_features.csv

Improved Model Performance:
R² score for forecasted_value_year_1: 0.5160

R² score for forecasted_value_year_2: 0.6263

R² score for forecasted_value_year_3: 0.5501


