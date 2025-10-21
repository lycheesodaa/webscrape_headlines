import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from arch import arch_model
import os

# Load Data
# Replace with your actual file path and column name
file_path = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/target_monthly/1-historical_price-Monthly.xlsx"
result_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/results/compare_models_monthly/GARCH"
data = pd.read_excel(file_path, engine="openpyxl")
data['Month-Year'] = pd.to_datetime(data['Month-Year'], format='%b-%Y')
data.set_index('Month-Year', inplace=True)

# Select the column for carbon credit prices
price_data = data['Price']

# Train-Test Split
train_ratio = 0.8
train_size = int(len(price_data) * train_ratio)
train_data = price_data[:train_size]
test_data = price_data[train_size:]

# Fit ARIMA for the Mean Model
p, d, q = 1, 1, 1  # ARIMA order
arima_model = ARIMA(train_data, order=(p, d, q))
arima_fit = arima_model.fit()

# Get ARIMA Residuals
residuals = arima_fit.resid

# Fit GARCH Model on ARIMA Residuals
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp="off")
print(garch_fit.summary())

# Multi-Step Ahead Forecasting
forecast_steps = len(test_data)  # Number of steps to forecast
arima_forecast = arima_fit.forecast(steps=forecast_steps)
mean_forecast = arima_forecast  # Mean forecast from ARIMA

# GARCH Variance Forecast
garch_forecast = garch_fit.forecast(horizon=forecast_steps)
variance_forecast = garch_forecast.variance.iloc[-1].values  # Variance forecasts

# Combine ARIMA and GARCH Forecasts
std_forecast = np.sqrt(variance_forecast)  # Convert variance to standard deviation
lower_bound = mean_forecast - 1.96 * std_forecast
upper_bound = mean_forecast + 1.96 * std_forecast

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(price_data.index, price_data, label="Historical Prices", color="blue")
plt.plot(test_data.index, mean_forecast, label="Forecasted Mean (ARIMA+GARCH)", color="orange")
plt.fill_between(test_data.index, lower_bound, upper_bound, color="gray", alpha=0.3, label="Confidence Interval")
plt.title("Carbon Credit Price Forecast with ARIMA+GARCH")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate Forecast
mape = mean_absolute_percentage_error(test_data, mean_forecast) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Save Results to Excel
forecast_df = pd.DataFrame({
    "True_Price": test_data,
    "Mean_Forecast": mean_forecast,
    "Lower_Bound": lower_bound,
    "Upper_Bound": upper_bound
}, index=test_data.index)

forecast_file = "arima_garch_forecast.xlsx"
forecast_df.to_excel(os.path.join(result_folder,forecast_file), engine="openpyxl")
print(f"Forecast results saved to {os.path.join(result_folder,forecast_file)}")
