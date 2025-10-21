import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
from sklearn.metrics import mean_absolute_percentage_error

# Load Data
file_path = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/target_monthly/1-historical_price-Monthly.xlsx"
result_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/results/compare_models_monthly/ARIMA"
data = pd.read_excel(file_path, engine="openpyxl")
data['Month-Year'] = pd.to_datetime(data['Month-Year'], format='%b-%Y')
data.set_index('Month-Year', inplace=True)

# Select the column for carbon credit prices
price_data0 = data['Price']
len_train_data = int(len(price_data0) * 0.8)

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(price_data0, label="Carbon Credit Price")
plt.title("Carbon Credit Price Over Time")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

price_data = price_data0[:len_train_data]

# Check for stationarity using Augmented Dickey-Fuller Test
adf_test = adfuller(price_data)
print(f"ADF Statistic: {adf_test[0]:.4f}")
print(f"p-value: {adf_test[1]:.4f}")
if adf_test[1] > 0.05:
    print("The series is not stationary. Differencing is required.")
else:
    print("The series is stationary.")

# Perform Differencing if Needed
if adf_test[1] > 0.05:
    diff_data = price_data.diff().dropna()
else:
    diff_data = price_data

# Plot ACF and PACF to Identify ARIMA Parameters
plot_acf(diff_data, lags=20)
plot_pacf(diff_data, lags=20)
plt.show()

# Fit ARIMA Model
p, d, q = 7, 1, 1  # Initial guess for ARIMA parameters
arima_model = ARIMA(price_data, order=(p, d, q))
arima_fit = arima_model.fit()

# Print Model Summary
print(arima_fit.summary())

# Iterative Forecasting
forecast_steps = len(price_data0) - len_train_data  # Number of steps to forecast
forecast_values = []
history = list(price_data)  # Start with the training data

for step in range(forecast_steps):
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    next_forecast = model_fit.forecast(steps=1)  # Forecast the next step
    forecast_values.append(next_forecast[0])
    history.append(next_forecast[0])  # Add the forecasted value to history for the next iteration

# Create Forecast Index
forecast_index = pd.date_range(price_data.index[-1], periods=forecast_steps + 1, freq='MS')[1:]

# Plot Forecasts
plt.figure(figsize=(10, 6))
plt.plot(price_data0, label="Historical Prices")
plt.plot(forecast_index, forecast_values, label="Forecast", color="orange")
plt.title("Iterative Carbon Credit Price Forecast with ARIMA")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

plot_file = os.path.join(result_folder, "Iterative_ARIMA_multistep.png")
plt.savefig(plot_file, dpi=300)

# Save Forecasts to Excel
forecast_df = pd.DataFrame({
    "Forecast": forecast_values
}, index=forecast_index)
forecast_file = os.path.join(result_folder, "iterative_arima_forecast.xlsx")
forecast_df.to_excel(forecast_file, engine="openpyxl")
print(f"Forecast saved to {forecast_file}")


test_mape = mean_absolute_percentage_error(price_data0[-forecast_steps:], forecast_values) * 100
print("test_mape=", test_mape)
