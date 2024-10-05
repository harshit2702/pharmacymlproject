import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from statsmodels.tsa.arima.model import ARIMA

# Load the CSV file
file_path = '/workspaces/codespaces-jupyter/Medical Inventory Optimization Dataset - Cleaned.csv'
data = pd.read_csv(file_path)

# Convert Dateofbill to datetime
data['Dateofbill'] = pd.to_datetime(data['Dateofbill'])

# Aggregate Quantity and Final_Sales by DrugName, Dateofbill, Specialisation, and Dept
aggregated_data = data.groupby(['DrugName', 'Dateofbill', 'Specialisation', 'Dept']).agg({
    'Quantity': 'sum',
    'Final_Sales': 'sum'
}).reset_index()

# Feature Engineering
# Extract date features
aggregated_data['Month'] = aggregated_data['Dateofbill'].dt.month
aggregated_data['Day'] = aggregated_data['Dateofbill'].dt.day
aggregated_data['Year'] = aggregated_data['Dateofbill'].dt.year
aggregated_data['Is_Weekend'] = aggregated_data['Dateofbill'].dt.dayofweek >= 5

# Cyclic features for Month
aggregated_data['Month_sin'] = np.sin(2 * np.pi * aggregated_data['Month'] / 12)
aggregated_data['Month_cos'] = np.cos(2 * np.pi * aggregated_data['Month'] / 12)

# Lag features for Quantity and Final_Sales
aggregated_data['Lag_7'] = aggregated_data.groupby('DrugName')['Final_Sales'].shift(7)

# Calculate rolling averages for Quantity and Final_Sales
aggregated_data['Quantity_MA'] = aggregated_data.groupby('DrugName')['Quantity'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
aggregated_data['Final_Sales_MA'] = aggregated_data.groupby('DrugName')['Final_Sales'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Drop NaN values after creating lag features
aggregated_data.dropna(inplace=True)

# Prepare data for modeling
X = aggregated_data[['Month_sin', 'Month_cos', 'Day', 'Year', 'Is_Weekend', 'Lag_7', 'Quantity_MA', 'Final_Sales_MA']]
y = aggregated_data['Final_Sales']

# Split data into training and testing sets using TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# 1. Linear Regression with Ridge Regularization
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_predictions, squared=False)

# 2. Random Forest with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
rf_predictions = best_rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)

# 3. LSTM Model
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled = X_scaled[train_index], X_scaled[test_index]

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential()
lstm_model.add(Input(shape=(1, X_train_scaled.shape[1])))
lstm_model.add(LSTM(50, return_sequences=True))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)
lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_rmse = mean_squared_error(y_test, lstm_predictions, squared=False)

# 4. Stacking Regressor (Combining Ridge and Random Forest)
estimators = [
    ('rf', best_rf_model),
    ('ridge', ridge_model)
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)
stacking_predictions = stacking_model.predict(X_test)
stacking_rmse = mean_squared_error(y_test, stacking_predictions, squared=False)

# Print RMSE for each model
print(f'Ridge Regression RMSE: {ridge_rmse}')
print(f'Random Forest RMSE (with hyperparameter tuning): {rf_rmse}')
print(f'LSTM RMSE: {lstm_rmse}')
print(f'Stacking Regressor RMSE: {stacking_rmse}')

# Calculate last known moving averages and lag for future predictions
last_known_date = aggregated_data['Dateofbill'].max()
last_known_quantity_ma = aggregated_data.loc[aggregated_data['Dateofbill'] == last_known_date, 'Quantity_MA'].values[0]
last_known_final_sales_ma = aggregated_data.loc[aggregated_data['Dateofbill'] == last_known_date, 'Final_Sales_MA'].values[0]
last_known_lag_7 = aggregated_data.loc[aggregated_data['Dateofbill'] == last_known_date, 'Lag_7'].values[0]

# Predict future sales using the best model (e.g., Stacking)
future_dates = pd.date_range(start=last_known_date, periods=30, freq='D')
future_data = pd.DataFrame({
    'Dateofbill': future_dates,
    'Month_sin': np.sin(2 * np.pi * future_dates.month / 12),
    'Month_cos': np.cos(2 * np.pi * future_dates.month / 12),
    'Day': future_dates.day,
    'Year': future_dates.year,
    'Is_Weekend': future_dates.dayofweek >= 5,
    'Lag_7': last_known_lag_7,
    'Quantity_MA': last_known_quantity_ma,
    'Final_Sales_MA': last_known_final_sales_ma
})

# Scale and reshape future data for LSTM
future_X = future_data[['Month_sin', 'Month_cos', 'Day', 'Year', 'Is_Weekend', 'Lag_7', 'Quantity_MA', 'Final_Sales_MA']]
future_X_scaled = scaler.transform(future_X)
future_X_lstm = future_X_scaled.reshape((future_X_scaled.shape[0], 1, future_X_scaled.shape[1]))

future_predictions = stacking_model.predict(future_X)  # or use lstm_model for future prediction
optimal_stock_levels = future_predictions * 1.1  # Adding 10% buffer

# Display future predictions and optimal stock levels
future_data['Predicted_Sales'] = future_predictions
future_data['Optimal_Stock_Level'] = optimal_stock_levels

print(future_data[['Dateofbill', 'Predicted_Sales', 'Optimal_Stock_Level']])