# Extreme
Datathon 2025 : PS3 : Wealth Management Strategies

TASK 1 ( CLASSIFICATION )

This project analyzes financial and demographic data of 10,000 clients to predict their recommended investment strategy using machine learning techniques. The dataset contains information on clients' financial profiles, investment behaviors, goals, and literacy.

 Dataset
The project utilizes two datasets:

1. **Static Features**: Contains per-client financial and demographic details.
    - `client_id`
    - `age`
    - `gender`
    - `employment_status`
    - `annual_income`
    - `debt_to_income_ratio`
    - `financial_knowledge_score`
    - `investment_goals`
    - `risk_appetite`
    - `investment_horizon_years`
    - `dependents`
    - `preferred_asset_classes`
    - `savings_rate`
    - `net_worth`

2. **Target Variables**: Contains investment strategy recommendations and forecasted financial values.
    - `recommended_strategy`
    - `forecasted_value_year_1`
    - `forecasted_value_year_2`
    - `forecasted_value_year_3`

 Steps Performed

### 1. Data Preparation
- Load both datasets as pandas DataFrames.
- Merge them using an inner join on `client_id`.

### 2. Exploratory Data Analysis (EDA)
- Identify and separate categorical and numerical variables.
- Visualize relationships between each feature and the target variable (`recommended_strategy`).

### 3. Feature Engineering
- Remove clients with negative income and store their `client_id` in `deleted_clients_df`.
- Convert `age` into categorical groups: `young`, `middle_aged`, `senior`.
- Perform appropriate encoding for categorical variables:
  - One-Hot Encoding: `gender`, `employment_status`, `investment_goals`
  - Ordinal Encoding: `risk_appetite`, `age_category`
  - Frequency Encoding: `preferred_asset_classes`
  - Target Encoding: `recommended_strategy`

### 4. Model Training
- Prepare training data by selecting features and encoding categorical variables.
- Train a Random Forest Classifier to predict the `recommended_strategy`.
- Tune hyperparameters to improve model accuracy.

## Technologies Used
- Python
- Pandas
- Seaborn & Matplotlib (for visualization)
- Scikit-learn (for preprocessing and modeling)
- XGBoost, LightGBM, CatBoost (for alternative modeling approaches)

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/financial-strategy-prediction.git
   cd financial-strategy-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to process the data and train models.

## Future Enhancements
- Implement advanced feature selection techniques.
- Experiment with deep learning models for better prediction accuracy.
- Develop a web application for real-time client strategy recommendations.

## Authors
Your Name (@your_github_handle)

## License
This project is licensed under the MIT License.





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


