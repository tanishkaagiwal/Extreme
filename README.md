# Extreme Datathon 2025: PS3 - Wealth Management Strategies

## TASK 1 ( CLASSIFICATION )

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


## Task 2 (Regression)

### Objective

To forecast each client's **portfolio_value** for the next 3 years using their 36-month historical time series and static attributes.

---

### Approach

#### 1. EDA & Preprocessing
- Merged **static_client_data** and **time_series_data**.
- Performed **trend analysis**, **autocorrelation checks**, and identified **non-stationary patterns**.
- Observed that analyzing features globally caused temporal trends to cancel out. Hence, processed data **client-wise**.

#### 2. Stationarity & Transformation
- Applied **differencing** to achieve stationarity **client-wise**.
- Engineered temporal features including:
  - **Lag values**.
  - **Rolling statistics** (mean and standard deviation over 3 and 6 months).

#### 3. Feature Selection
- Used **Extra Trees Regressor** to select the most impactful features from both static and dynamic datasets.

#### 4. Modeling
- Implemented an **LSTM model** trained on the selected features.

---

### Initial Results

- **R² score for forecasted_value_year_1**: 0.4745
- **R² score for forecasted_value_year_2**: 0.5614
- **R² score for forecasted_value_year_3**: 0.4653

---

### Feature Engineering for Final Model

Introduced intelligent features to enhance predictive power:

1. **portfolio_growth_rate**:
   ```python
   df.groupby('client_id')['portfolio_value_stationary'].pct_change()


## Future Enhancements
- Implement advanced feature selection techniques.
- Experiment with deep learning models for better prediction accuracy.
- Develop a web application for real-time client strategy recommendations.


## Task 3 (Stress testing)

This project adjusts the forecasted portfolio values of clients based on macroeconomic scenarios. It considers individual investment goals, risk appetite, and preferred asset classes to compute scenario-based adjusted portfolio values. The adjusted data is then used as input for a regression model (from Task 2) to make final predictions.

Data Description
1. Client Data (clients.csv)
This file contains static information about clients along with their forecasted portfolio values for three years. The columns include:

client_id: Unique identifier for the client

age, gender, employment_status: Basic demographic and employment details

annual_income: Yearly income of the client

debt_to_income_ratio: Ratio of debt to income

financial_knowledge_score: Score indicating financial literacy

investment_goals: Objective of investments (e.g., growth, stability)

risk_appetite: Risk tolerance level (Low, Medium, High)

investment_horizon_years: Number of years planned for investment

dependents: Number of financial dependents

preferred_asset_classes: Asset classes preferred by the client (stocks, bonds, real estate, etc.)

savings_rate: Percentage of income saved

net_worth: Total net worth of the client

forecasted_value_year_1, forecasted_value_year_2, forecasted_value_year_3: Expected portfolio value for the next three years

2. Macroeconomic Scenario Data (macro_scenarios.csv)
This file contains 10 different macroeconomic conditions, each described by multiple indicators:

scenario_id: Unique identifier for the macroeconomic scenario

interest_rate_change: Change in interest rates

inflation_spike: Indicator for inflation increase

market_volatility_shock: Boolean indicating market instability

equity_impact: Impact on stock markets

fixed_income_impact: Impact on bond markets

macroeconomic_score_adjustment: Overall economic performance indicator

sentiment_index_adjustment: Market sentiment effect

3. Adjusted Portfolio Data (adjusted_portfolio_scenario_1.csv, adjusted_portfolio_scenario_2.csv, etc.)
Each file contains the adjusted forecasted values for all clients under a specific scenario. The structure is similar to clients.csv, but with adjusted values for each scenario.

Methodology
Step 1: Load Data
Read clients.csv for static client information and macro_scenarios.csv for economic conditions.

Step 2: Apply Scenario-Based Adjustments
Each client's preferred asset classes determine how different macroeconomic factors affect their portfolio.

A risk multiplier (Low = 0.8, Medium = 1.0, High = 1.2) adjusts the impact of economic conditions.

Portfolio values are modified based on the formula:

Adjusted Value
=
Forecasted Value
×
(
1
+
Total Impact
×
Risk Multiplier
)
Adjusted Value=Forecasted Value×(1+Total Impact×Risk Multiplier)
Step 3: Generate Scenario-Specific Outputs
A separate CSV file (adjusted_portfolio_scenario_X.csv) is generated for each macroeconomic scenario.

Step 4: Regression Model Input
The adjusted portfolio data for Scenario 1 is stored in prediction.csv, which serves as input for the regression model from Task 2.

How to Run the Code
Prerequisites
Python 3.x

Required libraries: pandas, numpy

Execution Steps
Place clients.csv and macro_scenarios.csv in the working directory.

Run the script:

bash
Copy
Edit
python adjust_portfolio.py
Outputs:

adjusted_portfolio_scenario_1.csv to adjusted_portfolio_scenario_10.csv (one for each scenario).

prediction.csv (used for regression model input).

Next Steps
Use prediction.csv as input for the regression model from Task 2.

Train the model and generate portfolio value predictions for future year





