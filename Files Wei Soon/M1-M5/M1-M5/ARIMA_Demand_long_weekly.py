from itertools import product
from typing import Union, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import matplotlib.dates as mdates
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
result_folder = Path("/home/lianlian/webscrape_headlines/results/ARIMA_demand_long_weekly/")
instance_norm = False
if instance_norm:
    result_folder = result_folder / 'instance_norm'
result_folder.mkdir(exist_ok=True, parents=True)

# Read and prepare data
data = pd.read_csv(os.path.join(data_folder, 'demand_full_weekly.csv'), index_col=0, parse_dates=True)
print(data.columns)
demand_data = data['system_demand_actual']

# Parameters
window_sizes = [512]
horizons = [336]  # Predict X days ahead
p, d, q, s = 1, 1, 1, 52  # ARIMA parameters
max_p, max_q = 5, 5


def create_timestamps(
        last_timestamp: Union[datetime.datetime, pd.Timestamp],
        freq: Optional[Union[int, float, datetime.timedelta, pd.Timedelta, str]] = None,
        time_sequence: Optional[Union[List[int], List[float], List[datetime.datetime], List[pd.Timestamp]]] = None,
        periods: int = 1,
) -> List[pd.Timestamp]:
    """Simple utility to create a list of timestamps based on start, delta and number of periods

    Args:
        last_timestamp (Union[datetime.datetime, pd.Timestamp]): The last observed timestamp, new timestamps will be created
            after this timestamp.
        freq (Optional[Union[int, float, datetime.timedelta, pd.Timedelta, str]], optional): The frequency at which timestamps
            should be generated. Defaults to None.
        time_sequence (Optional[Union[List[int], List[float], List[datetime.datetime], List[pd.Timestamp]]], optional): A time sequence
            from which the frequency can be inferred. Defaults to None.
        periods (int, optional): The number of timestamps to generate. Defaults to 1.

    Raises:
        ValueError: If the frequency cannot be parsed from freq or inferred from time_sequence

    Returns:
        List[pd.Timestamp]: List of timestamps
    """

    if freq is None and time_sequence is None:
        raise ValueError("Neither `freq` nor `time_sequence` provided, cannot determine frequency.")

    if freq is None:
        # to do: make more robust
        freq = time_sequence[-1] - time_sequence[-2]

    # more complex logic is required to support all edge cases
    if isinstance(freq, (pd.Timedelta, datetime.timedelta, str)):
        try:
            # try date range directly
            return pd.date_range(
                last_timestamp,
                freq=freq,
                periods=periods + 1,
            ).tolist()[1:]
        except ValueError as e:
            # if it fails, we can try to compute a timedelta from the provided string
            if isinstance(freq, str):
                freq = pd._libs.tslibs.timedeltas.Timedelta(freq)
                return pd.date_range(
                    last_timestamp,
                    freq=freq,
                    periods=periods + 1,
                ).tolist()[1:]
            else:
                raise e
    else:
        # numerical timestamp column
        return [last_timestamp + i * freq for i in range(1, periods + 1)]


mapes_dfs = pd.DataFrame()
for window_size in window_sizes:
    print(f'Naive best order for ARIMA (CTX{window_size}): ({p}, {d}, {q})')
    mapes = []

    for horizon in horizons:
        # Initialize arrays to store predictions and actual values
        predictions = []
        actuals = []
        dates = []

        # Extract window - use X months before the prediction point
        train_window = demand_data[len(data) - window_size:len(data)]

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

        predictions = np.array(predictions)
        output_df = pd.DataFrame({
            'system_demand_actual': predictions
        })

        output_df['datetime'] = create_timestamps(last_timestamp=data.index[-1], freq='W',
                                                  periods=horizon)

        # dset_test should be the same size as reshaped
        output_df.to_csv(result_folder / f'ARIMA_ctx{window_size}_pl{horizon}_zeroshot.csv')


        # plot and export figure
        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data, x='datetime', y='system_demand_actual', label='true')
        sns.lineplot(output_df, x='datetime', y='system_demand_actual', label='pred')

        ax = plt.gca()
        locator = mdates.AutoDateLocator(minticks=10, maxticks=12)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        plt.xlabel('Date')
        plt.ylabel('Demand (KwH)')
        plt.title('5-year ahead weekly demand forecasts (ARIMA)')
        plt.legend()
        plt.grid()
        plt.show()
        plt.tight_layout()
        fig.savefig(result_folder / 'arima_5year.png', bbox_inches='tight')
