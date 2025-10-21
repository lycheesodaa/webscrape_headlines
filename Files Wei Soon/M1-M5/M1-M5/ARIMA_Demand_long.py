from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error

# Explicitly ignore arima warning messages
import warnings
warnings.filterwarnings("ignore")

# Load Data
data_folder = "/home/lianlian/models/Time-LLM/dataset/demand/"
result_folder = Path("/home/lianlian/webscrape_headlines/results/ARIMA_demand_long/")
instance_norm = False
if instance_norm:
    result_folder = result_folder / 'instance_norm'
result_folder.mkdir(exist_ok=True, parents=True)

# Read and prepare data
data = pd.read_csv(os.path.join(data_folder, 'demand_full_daily.csv'), index_col=0, parse_dates=True)
print(data.columns)
demand_data = data['system_demand_actual']

# Parameters
window_sizes = [1024, 1536]
horizons = [720]  # Predict X days ahead
p, d, q, s = 1, 1, 1, 7  # ARIMA parameters
max_p, max_q = 5, 5

# Calculate the start of test period (last 20% of data)
test_size = int(len(demand_data) * 0.2)
test_start_idx = len(demand_data) - test_size - 1


'''
Naively find the optimal ARIMA order for p and q with a subsample of the context windows.
Sampled context windows are spaced `win_size` apart.
'''
def find_optimal_params(win_size):
    windows = []
    for idx in range(win_size, len(demand_data), win_size):
        windows.append(demand_data[idx - win_size:idx])

    best_aic = float('inf')
    best_order = None

    for p, q in product(range(max_p + 1), range(max_q + 1)):
        aic_list = []
        for window in windows:
            try:
                model = ARIMA(window, order=(p, d, q))
                aic_list.append(model.fit().aic)
            except:
                continue

        if np.mean(aic_list) < best_aic:
            best_aic = np.mean(aic_list)
            best_order = (p, d, q)

    return best_order


mapes_dfs = pd.DataFrame()
for window_size in window_sizes:
    # p, d, q = find_optimal_params(window_size)
    print(f'Naive best order for ARIMA (CTX{window_size}): ({p}, {d}, {q})')
    mapes = []

    for horizon in horizons:
        # Initialize arrays to store predictions and actual values
        predictions = []
        actuals = []
        dates = []

        # Rolling window prediction for last 20% of data
        for i in range(test_start_idx, len(demand_data) - horizon + 1):
            # Extract window - use X months before the prediction point
            train_window = demand_data[i - window_size:i]

            if instance_norm:
                scaler = StandardScaler()
                train_window = scaler.fit_transform(train_window.values.reshape(-1, 1)).ravel()

            # Fit model on window
            model = ARIMA(train_window, order=(p, d, q), seasonal_order=(p, d, q, s))
            model_fit = model.fit()

            # Make prediction
            forecast = model_fit.forecast(steps=horizon)
            predictions.extend(forecast)

            if instance_norm:
                mean_forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).reshape(-1)

            # Store actual value
            actual = demand_data[i:i + horizon]
            actuals.extend(actual)
            assert len(forecast) == len(actual)
            dates.extend(demand_data.index[i:i + horizon])

        # Convert to arrays for easier handling
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        dates = pd.DatetimeIndex(dates)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(demand_data.index, demand_data, label='Historical Data', alpha=0.7)
        for i in range(len(predictions) // horizon):
            plt.plot(dates[i*horizon:i*horizon+horizon], predictions[i*horizon:i*horizon+horizon], linestyle='--')
        plt.axvline(x=demand_data.index[test_start_idx], color='r', linestyle=':', label='Test Period Start')
        plt.title(f'Rolling Window ARIMA Predictions\nWindow Size: {window_size} days, Last 20% of data')
        plt.xlabel('Time')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_file = os.path.join(result_folder, f"Rolling_ARIMA_w{window_size}_pl{horizon}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()

        # Save predictions to DataFrame
        results_df = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions,
            'AbsoluteError': np.abs(actuals - predictions),
            'PercentageError': np.abs((actuals - predictions) / actuals) * 100
        }, index=dates)

        # Calculate metrics
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        print(f"Rolling Window MAPE: {mape:.2f}%")
        mapes.append(mape)

        # Save to Excel
        excel_file = os.path.join(result_folder, f"rolling_arima_w{window_size}_pl{horizon}.xlsx")
        results_df.to_excel(excel_file)
        print(f"Results saved to {excel_file}")

        # Additional analysis
        print("Error Analysis:")
        print(f"Mean Absolute Error: {np.mean(np.abs(actuals - predictions)):.2f}")
        print(f"Mean Percentage Error: {np.mean((predictions - actuals) / actuals * 100):.2f}%")
        print(f"Standard Deviation of Absolute Error: {np.std(np.abs(actuals - predictions)):.2f}\n")

        # # Plot error distribution
        # plt.figure(figsize=(10, 5))
        # plt.hist(predictions - actuals, bins=30, edgecolor='black')
        # plt.title('Distribution of Prediction Errors')
        # plt.xlabel('Error')
        # plt.ylabel('Frequency')
        # error_plot_file = os.path.join(result_folder, f"Error_Distribution_w{window_size}_pl{horizon}.png")
        # plt.savefig(error_plot_file, dpi=300)
        # plt.close()
        #
        # # Plot actual vs predicted
        # plt.figure(figsize=(10, 5))
        # plt.scatter(actuals, predictions, alpha=0.5)
        # plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')  # Perfect prediction line
        # plt.title('Actual vs Predicted Values')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # scatter_plot_file = os.path.join(result_folder, f"Actual_vs_Predicted_w{window_size}_pl{horizon}.png")
        # plt.savefig(scatter_plot_file, dpi=300)
        # plt.close()

    mapes_dfs["MAPE_ctx"+str(window_size)] = mapes

mapes_dfs.to_csv(os.path.join(result_folder,"mapes_lookbacks_multiSteps.csv"))
print(mapes_dfs)