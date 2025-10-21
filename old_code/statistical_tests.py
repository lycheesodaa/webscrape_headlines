import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
# import matplotlib.pyplot as plt
# import numpy as np
from datetime import datetime, timedelta

from statsforecast import StatsForecast
from tqdm import tqdm
from statsforecast.models import AutoARIMA, GARCH, ARCH
from utilsforecast.losses import mse, mape
from utilsforecast.evaluation import evaluate


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
data = data[['datetime', 'actual']]
data.rename(columns={'datetime': 'ds', 'actual': 'y'}, inplace=True)

test_size = int(len(data) * 0.2)  # val size = test size

context_length = 512
pred_lens = [12, 24, 36, 48, 60, 72]

all_preds = {pred_len: [] for pred_len in pred_lens}
# all_conf_ints = {pred_len: [] for pred_len in pred_lens}
dates = {pred_len: [] for pred_len in pred_lens}

models = [AutoARIMA(),
          ARCH(1),
          ARCH(2),
          GARCH(1, 1),
          GARCH(1, 2),
          GARCH(2, 2),
          GARCH(2, 1), ]
sf = StatsForecast(
    models=models,
    freq='h',
    n_jobs=-1,
    verbose=True
)


def evaluate_cross_validation(df, metric):
    df = df.reset_index()
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y']).columns.tolist()
    evals = []
    # Calculate loss for every unique_id and cutoff.
    for cutoff in df['cutoff'].unique():
        eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=[metric], models=models)
        evals.append(eval_)
    evals = pd.concat(evals)
    # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals = evals.groupby('unique_id').mean(numeric_only=True)
    evals['best_model'] = evals.idxmin(axis=1)
    return evals


# #############################################################################

print("Predicting...")
eval_dfs = []
for pred_len in tqdm(pred_lens):
    data['unique_id'] = f'pl{pred_len}'
    cross_val_df = sf.cross_validation(
        df=data,
        h=pred_len,
        input_size=context_length,
        step_size=1,
        n_windows=test_size - pred_len,
        # n_windows=1,
    )
    # print(cross_val_df)

    mse_eval = evaluate_cross_validation(cross_val_df, mse)
    mse_eval.rename(index={mse_eval.index[0]: f'{mse_eval.index[0]}_mse'}, inplace=True)
    eval_dfs.append(mse_eval)
    mape_eval = evaluate_cross_validation(cross_val_df, mape)
    mape_eval.rename(index={mape_eval.index[0]: f'{mape_eval.index[0]}_mape'}, inplace=True)
    eval_dfs.append(mape_eval)

    result_df = pd.concat(eval_dfs)
    result_df.to_csv(f'ARIMA_AUS_{pred_len}_losses.csv')

print("Done.")
