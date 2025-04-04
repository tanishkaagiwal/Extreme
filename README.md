# Extreme Datathon 2025: PS3 - Wealth Management Strategies

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








