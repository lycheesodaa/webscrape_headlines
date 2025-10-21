import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
# import matplotlib.pyplot as plt
# import numpy as np
from datetime import datetime, timedelta

from statsforecast import StatsForecast
from tqdm import tqdm
from statsforecast.models import AutoARIMA


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
# data = pd.read_csv('external_data/demand/demand_data_all_cleaned_numerical.csv')
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
# all_conf_ints = {pred_len: [] for pred_len in pred_lens}
dates = {pred_len: [] for pred_len in pred_lens}
start_datetime = train['datetime'].iloc[-1] + timedelta(minutes=60)


# #############################################################################

print("Predicting...")
# using range(len(test)) will cause horizons > 1 to predict more than the required number of horizons
# just truncate the extra in post-processing
arima = pm.auto_arima(train_y, suppress_warnings=True, maxiter=30,
                      stepwise=True, error_action='ignore', trace=True)
# arima = pm.auto_arima(train_y, m=24, maxiter=30, method='nm', suppress_warnings=True,
#                       stepwise=True, error_action='ignore', trace=True)
for window_num in tqdm(range(len(test))):
    # setting a constant context length
    # if window_num < context_length:
    #     context = np.concatenate([train_y[-context_length + window_num:], test_y[:window_num]])
    # else:
    #     context = test_y[window_num - context_length:window_num]

    # best model is (3, 0, 0) (0, 0, 2, 24) but we use the below because its 3x faster with marginally lower perf.
    # arima = pm.ARIMA(order=(3, 0, 0), seasonal_order=(0, 0, 2, 24), suppress_warnings=True)
    # arima.fit(context)

    for pred_len in pred_lens:
        preds = arima.predict(n_periods=pred_len,
                              return_conf_int=False)

        all_preds[pred_len].extend(preds)
        # all_conf_ints[pred_len].extend(conf_int)
        dates[pred_len].extend(generate_datetime_list(start_datetime, pred_len))

    start_datetime += timedelta(minutes=60)

    arima.update(test_y[window_num])

print("Exporting...")
for pred_len in pred_lens:
    df = pd.DataFrame({
        'date': dates[pred_len],
        'pred': all_preds[pred_len],
        # 'cis': all_conf_ints[pred_len],
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.to_csv(f'ARIMA_AUS_{pred_len}_predictions.csv')

print("Done.")
