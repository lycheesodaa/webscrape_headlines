from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from arch import arch_model
import os

# Explicitly ignore arima warning messages
import warnings
warnings.filterwarnings("ignore")

# Load Data
data_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/"
result_folder = Path("/home/lianlian/webscrape_headlines/results/compare_models_daily/GARCH_rolling")
instance_norm = True
if instance_norm:
    result_folder = result_folder / 'instance_norm'
result_folder.mkdir(exist_ok=True, parents=True)

# Read and prepare data
data = pd.read_csv(os.path.join(data_folder, 'res_daily/merged_data_imputed.csv'))
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)  # Use 'Month-Year' as the index
price_data = data['Price']

# Parameters
window_sizes = [512]  # 20 months window
# horizons = [1, 2, 3, 4, 5, 7, 14, 21, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 180, 365, 545]  # Predict X days ahead
horizons = [180]  # Predict X days ahead
p, d, q = 1, 1, 1  # ARIMA order

# Calculate the start of test period (last 20% of data)
test_size = int(len(price_data) * 0.2)
test_start_idx = len(price_data) - test_size

mapes_dfs = pd.DataFrame()
for horizon in horizons:
    mapes = []

    for window_size in window_sizes:
        # Initialize arrays for storing results
        predictions = []
        lower_bounds = []
        upper_bounds = []
        actuals = []
        dates = []

        # Rolling window prediction for last 20% of data
        for i in range(test_start_idx, len(price_data) - horizon):
            # Extract window
            train_window = price_data[i - window_size:i]

            if instance_norm:
                scaler = StandardScaler()
                train_window = scaler.fit_transform(train_window.values.reshape(-1, 1)).ravel()

            try:
                # Fit ARIMA
                arima_model = ARIMA(train_window, order=(p, d, q))
                arima_fit = arima_model.fit()

                # Get ARIMA residuals
                residuals = arima_fit.resid

                # Fit GARCH on residuals
                garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
                garch_fit = garch_model.fit(disp="off")

                # Get forecasts
                mean_forecast = arima_fit.forecast(steps=horizon)
                variance_forecast = garch_fit.forecast(horizon=horizon).variance.iloc[-1].values

                if instance_norm:
                    mean_forecast = scaler.inverse_transform(mean_forecast.reshape(-1, 1)).reshape(-1)

                # Calculate prediction intervals
                std_forecast = np.sqrt(variance_forecast)
                lower_bound = mean_forecast - 1.96 * std_forecast
                upper_bound = mean_forecast + 1.96 * std_forecast

                # Store results
                predictions.extend(mean_forecast)
                lower_bounds.extend(lower_bound)
                upper_bounds.extend(upper_bound)
                actuals.extend(price_data[i:i + horizon])
                dates.extend(price_data.index[i:i + horizon])

            except Exception as e:
                print(f"Error at index {i}: {str(e)}")
                continue

        # Convert to arrays
        predictions = np.array(predictions)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        actuals = np.array(actuals)
        dates = pd.DatetimeIndex(dates)

        # Calculate metrics
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        print(f"Rolling Window MAPE: {mape:.2f}%")
        mapes.append(mape)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(price_data.index, price_data, label='Historical Data', alpha=0.7)
        for i in range(len(predictions) // horizon):
            plt.plot(dates[i*horizon:i*horizon+horizon], predictions[i*horizon:i*horizon+horizon], linestyle='--')
            plt.fill_between(dates[i*horizon:i*horizon+horizon],
                             lower_bounds[i*horizon:i*horizon+horizon],
                             upper_bounds[i*horizon:i*horizon+horizon],
                             color=plt.gca().lines[-1].get_color(), alpha=0.3)
        plt.axvline(x=price_data.index[test_start_idx], color='r', linestyle=':', label='Test Period Start')
        plt.title(f'Rolling Window ARIMA+GARCH Predictions\nWindow Size: {window_size} months, Last 20% of data')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_file = os.path.join(result_folder, f"Rolling_ARIMA_GARCH_w{window_size}_pl{horizon}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()

        # Save predictions to DataFrame
        results_df = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions,
            'Lower_Bound': lower_bounds,
            'Upper_Bound': upper_bounds,
            'AbsoluteError': np.abs(actuals - predictions),
            'PercentageError': np.abs((actuals - predictions) / actuals) * 100
        }, index=dates)

        # Save to Excel
        excel_file = os.path.join(result_folder, f"rolling_arima_garch_w{window_size}_pl{horizon}.xlsx")
        results_df.to_excel(excel_file)
        print(f"Results saved to {excel_file}")

        # Additional analysis
        print("Error Analysis:")
        print(f"Mean Absolute Error: {np.mean(np.abs(actuals - predictions)):.2f}")
        print(f"Mean Percentage Error: {np.mean((predictions - actuals) / actuals * 100):.2f}%")
        print(f"Standard Deviation of Absolute Error: {np.std(np.abs(actuals - predictions)):.2f}")

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

        # Calculate coverage of prediction intervals
        within_bounds = np.sum((actuals >= lower_bounds) & (actuals <= upper_bounds))
        coverage = within_bounds / len(actuals) * 100
        print(f"Prediction Interval Coverage: {coverage:.2f}%\n")

    mapes_dfs["MAPE_step"+str(horizon)] = mapes

mapes_dfs.to_csv(os.path.join(result_folder,"mapes_lookbacks_multiSteps.csv"))
print(mapes_dfs)