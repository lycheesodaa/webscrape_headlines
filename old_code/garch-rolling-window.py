import numpy as np
import pandas as pd
from arch import arch_model
from pmdarima import model_selection
from datetime import datetime, timedelta

from tqdm import tqdm


def generate_datetime_list(start_datetime_: datetime, num_intervals: int) -> list:
    """
    Generate a list of datetimes in 30-minute intervals.

    Args:
    start_datetime_ (datetime): The starting datetime.
    num_intervals (int): The number of intervals to generate.

    Returns:
    list: A list of datetimes.
    """
    interval = timedelta(minutes=60)
    return [start_datetime_ + i * interval for i in range(num_intervals)]

# #############################################################################
# Load the data and split it into separate pieces
data = pd.read_csv('external_data/demand/demand_data_all_nsw_numerical.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
assert pd.infer_freq(data['datetime']) == 'h'

# 0.8 - 0.2 split
train, test = model_selection.train_test_split(data, train_size=0.8)
train_y = train['actual'].values
test_y = test['actual'].values
val_size = len(test)  # val size = test size

context_length = 512
pred_lens = [1, 12, 24, 36, 48, 60, 72]

all_preds = {pred_len: [] for pred_len in pred_lens}
dates = {pred_len: [] for pred_len in pred_lens}
start_datetime = train['datetime'].iloc[-1] + timedelta(minutes=60)


# #############################################################################

print("Predicting...")
# using range(len(test)) will cause horizons > 1 to predict more than the required number of horizons
# just truncate the extra in post-processing
for window_num in tqdm(range(len(test))):
    # setting a constant context length
    if window_num < context_length:
        context = np.concatenate([train_y[-context_length + window_num:], test_y[:window_num]])
    else:
        context = test_y[window_num - context_length:window_num].copy()

    assert len(context) == context_length
    context *= 0.01  # rescaling the inputs for better parameter estimation

    model = arch_model(context, vol='GARCH', p=1, q=1)
    results_fit = model.fit(disp='off')

    # Forecast volatility for multiple horizons
    forecast = results_fit.forecast(horizon=max(pred_lens))

    # Forecast volatility for multiple horizons
    for pred_len in pred_lens:
        # Extract the forecasted variance for the specific horizon
        preds = forecast.mean.iloc[-1, :pred_len]
        preds /= 0.01  # unscaling the outputs TODO multiply garch outputs by 10

        all_preds[pred_len].extend(preds)
        dates[pred_len].extend(generate_datetime_list(start_datetime, pred_len))

    start_datetime += timedelta(minutes=60)

print("Exporting...")
for pred_len in pred_lens:
    df = pd.DataFrame({
        'date': dates[pred_len],
        'pred': all_preds[pred_len],
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.to_csv(f'results/GARCH_AUS_{pred_len}_predictions.csv')

print("Done.")
