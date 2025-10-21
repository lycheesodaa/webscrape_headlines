import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

# Paths
data_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/"
feature_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/features_daily/"  # Folder containing Excel files
target_file = "/home/lianlian/models/Time-LLM/dataset/carbon/target_daily/1-historical_price-Daily.xlsx"  # File with carbon credit price data
result_folder = Path("/home/lianlian/webscrape_headlines/results/compare_models_daily/SVR")
output_steps = [1, 2, 3, 4, 5, 7, 14, 21, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 180, 365, 545]  # Predict X days ahead
lookback = 512 # Number of past days to use as input
train_times = 1
fea_select_flag = 1 # choose with or without feature selection
TopFeaturePercs = [0.25, 0.5, 0.75] # Select top % of the full set of features
GPU_ID = 0
train_size = 0.6
val_size = 0.2
test_size = 0.2
instance_norm = True
if instance_norm:
    result_folder = result_folder / 'instance_norm'
result_folder.mkdir(parents=True, exist_ok=True)

data_df = pd.read_csv(os.path.join(data_folder, 'res_daily/merged_data_imputed.csv'))
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df.set_index('Date', inplace=True)  # Use 'Month-Year' as the index

# Prepare input
def create_sequences(X, y, lookback, output_steps):
    if not instance_norm:
        X_seq, y_seq, scalers = [], [], []
        for i in range(lookback, len(X) - output_steps + 1):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i:i + output_steps])
        return np.array(X_seq), np.array(y_seq), scalers  # dummy var so the func signature stays the same
    else:
        X_seq, y_seq, scalers = [], [], []

        for i in range(lookback, len(X) - output_steps + 1):
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_window = X[i - lookback:i]
            y_window = y[i:i + output_steps]

            X_normalized = scaler_X.fit_transform(X_window)

            scaler_y.fit(X_window[:, :1])
            y_normalized = scaler_y.transform(y_window.reshape(-1, 1))

            X_seq.append(X_normalized)
            y_seq.append(y_normalized)
            scalers.append(scaler_y)

        return np.array(X_seq), np.array(y_seq), scalers

mapes_dfs = pd.DataFrame()
for train_time in range(0, train_times):
    print('train time=', train_time)

    for TopFeaturePerc in TopFeaturePercs:
        print("TopFeaturePerc=", TopFeaturePerc)
        mapes = []

        # Load in selected features based on the spearman correlation analysis
        sel_features_df0 = pd.read_excel(os.path.join(data_folder, "res_daily/ranked_abs_features_daily.xlsx"))
        sel_feature_len = int(TopFeaturePerc*len(sel_features_df0))
        sel_features_df0.sort_values(by="Correlation", ascending=False, inplace=True)
        sel_feature_names = sel_features_df0["Factor"][0:sel_feature_len]
        sel_feature_names = [val for val in sel_feature_names if 'Historical' not in val]
        print("sel_feature_names=",sel_feature_names)
        print("sel_feature_length=",sel_feature_len)

        data = data_df[['Price'] + sel_feature_names]

        # Separate input features and target
        X = data.values  # All columns
        y = data.iloc[:, 0].values  # First column is the target price

        if not instance_norm:
            # Scale data
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y.reshape(-1, 1))

        for output_step in output_steps:
            print("output_step=", output_step)

            # Calculate data split indices
            train_length = int(len(X) * train_size)
            val_length = int(len(X) * val_size)
            test_length = int(len(X) * test_size)
            
            # Create data splits with lookback consideration
            train_end = train_length
            val_start = train_length - lookback  # Include lookback period from training
            val_end = train_length + val_length
            test_start = val_end - lookback      # Include lookback period from validation
            test_end = val_end + test_length
            
            # Split data into train/validation/test sets
            data_splits = {
                'train': (X[:train_end], y[:train_end]),
                'val': (X[val_start:val_end], y[val_start:val_end]),
                'test': (X[test_start:test_end], y[test_start:test_end])
            }
            
            # Create sequences for each split
            sequences = {}
            for split_name, (X_split, y_split) in data_splits.items():
                X_seq, y_seq, scalers = create_sequences(X_split, y_split, lookback, output_step)
                sequences[split_name] = (X_seq, y_seq, scalers)
            
            # Unpack sequences
            X_train, y_train, scalers_train = sequences['train']
            X_val, y_val, scalers_val = sequences['val']
            X_test, y_test, scalers_test = sequences['test']
            
            print(f"Training set: {X_train.shape}, {y_train.shape}")
            print(f"Validation set: {X_val.shape}, {y_val.shape}")
            print(f"Test set: {X_test.shape}, {y_test.shape}")

            # Multi-output SVR
            multi_svr = MultiOutputRegressor(SVR(kernel='rbf', C=1000, epsilon=0.01, verbose=True))
            
            # Use only training data for model training
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            
            # Train the model
            multi_svr.fit(X_train_flat, y_train_flat)

            # Make predictions on the test set
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred_scaled = multi_svr.predict(X_test_flat)
            y_test_flat = y_test.reshape(y_test.shape[0], -1)

            # Inverse scaling
            if instance_norm:
                y_pred_inverse = np.zeros_like(y_pred_scaled)
                y_test_inverse = np.zeros_like(y_test_flat)
                for i, pred in enumerate(y_pred_scaled):
                    y_pred_inverse[i] = scalers_test[i].inverse_transform(pred.reshape(1, -1)).reshape(-1)
                    y_test_inverse[i] = scalers_test[i].inverse_transform(y_test_flat[i].reshape(1, -1)).reshape(-1)
            else:
                y_pred_inverse = scaler_y.inverse_transform(y_pred_scaled)
                y_test_inverse = scaler_y.inverse_transform(y_test_flat)

            # Calculate MAPE
            mape = np.mean(np.abs((y_pred_inverse - y_test_inverse) / y_test_inverse)) * 100
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            mapes.append(mape)

            # Plot the Full Price Data and Predictions
            plt.figure(figsize=(15, 8))

            # Plot Historical Price Data
            plt.plot(data_df.index, data_df['Price'], label="Historical Prices", color="blue", linewidth=2)
            
            # Add vertical lines for train/val/test splits
            plt.axvline(x=data_df.index[train_length], color='r', linestyle=':', label='Train-Val Split')
            plt.axvline(x=data_df.index[val_end], color='g', linestyle=':', label='Val-Test Split')

            # Add Test Predictions to the Plot
            start_test_index = len(data_df) - len(y_test)
            for i, prediction in enumerate(y_pred_inverse):
                forecast_dates = [data_df.index[start_test_index + i] + pd.DateOffset(days=j) for j in
                                  range(output_step)]
                plt.plot(forecast_dates, prediction, color="red", marker="o", linestyle="-",
                         label="Predicted" if i == 0 else "")

            plt.title("Carbon Credit Price Multi-Step Forecasting with SVR")
            plt.xlabel("Time")
            plt.ylabel("Carbon Price")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.grid(True)
            plt.tight_layout()

            # Save Plot
            plot_file = os.path.join(result_folder, f"SVR_MultiStep_feat{TopFeaturePerc}_lb{lookback}_pl{output_step}.png")
            plt.savefig(plot_file, dpi=300)
            print(f"Plot saved to {plot_file}")
            # plt.show()
            plt.close()

            # Save Results to Excel
            results_df = pd.DataFrame(
                np.column_stack([
                    val for step in range(output_step)
                    for val in [y_test_inverse[:, step], y_pred_inverse[:, step]]
                ]),
                columns=[
                    item for step in range(output_step)
                    for item in [f"True_Step_{step + 1}", f"Predicted_Step_{step + 1}"]
                ]
            )
            results_file = os.path.join(result_folder, f"SVR_MultiStep_feat{TopFeaturePerc}_lb{lookback}_pl{output_step}.xlsx")
            results_df.to_excel(results_file, index=False, engine="openpyxl")
            print(f"Results saved to {results_file}")

        mapes_dfs["MAPE_feat"+str(TopFeaturePerc)+"_train"+str(train_time)] = mapes

mapes_dfs.to_csv(os.path.join(result_folder,"mapes_lookbacks_multiSteps.csv"))
# mapes_steps = mapes_dfs
print(mapes_dfs) 