import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
import matplotlib.pyplot as plt
import numpy as np
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
    interval = timedelta(minutes=30)
    return [start_datetime_ + i * interval for i in range(num_intervals)]

# #############################################################################
# Load the data and split it into separate pieces
data = pd.read_csv('~/models/MOMENT/data/demand_data_all_cleaned_numerical.csv')
data['datetime'] = pd.to_datetime(data['datetime'])

# 0.8 - 0.2 split
train, test = model_selection.train_test_split(data, train_size=0.8)
train_y = train['actual'].values
test_y = test['actual'].values
val_size = len(test)  # val size = test size

pred_lens = [1, 12, 72]

for pred_len in pred_lens:
    # #############################################################################
    # Fit with some validation (cv) samples
    arima = pm.auto_arima(train_y, start_p=1, start_q=1, d=0, max_p=5, max_q=5,
                          out_of_sample_size=val_size, suppress_warnings=True,
                          stepwise=True, error_action='ignore')

    all_preds = []
    all_conf_ints = []
    dates = []
    start_datetime = train['datetime'].iloc[-1] + timedelta(minutes=30)

    for window_num in tqdm(range(len(test) - pred_len + 1)):
        preds, conf_int = arima.predict(n_periods=pred_len,
                                        return_conf_int=True)
        all_preds.extend(preds)
        all_conf_ints.extend(conf_int)
        dates.extend(generate_datetime_list(start_datetime, pred_len))
        start_datetime += timedelta(minutes=30)

        arima.update(test_y[window_num])

    df = pd.DataFrame({
        'date': dates,
        'pred': all_preds,
        'cis': all_conf_ints,
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.to_csv(f'ARIMA_{pred_len}_predictions.csv')
