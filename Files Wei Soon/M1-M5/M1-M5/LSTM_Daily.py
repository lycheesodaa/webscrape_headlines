import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Paths
data_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/"
feature_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/features_daily/"  # Folder containing Excel files
target_file = "/home/lianlian/models/Time-LLM/dataset/carbon/target_daily/1-historical_price-Daily.xlsx"  # File with carbon credit price data
result_folder = Path("/home/lianlian/webscrape_headlines/results/compare_models_daily/LSTM")
output_steps = [1, 2, 3, 4, 5, 7, 14, 21, 28, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 180, 365, 545]  # Predict X days ahead
lookback = 512 # Number of past days to use as input
train_times = 3
fea_select_flag = 1 # choose with or without feature selection
TopFeaturePercs = [0, 0.25, 0.5, 0.75] # Select top % of the full set of features
GPU_ID = 1
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
        for i in range(len(X) - lookback - output_steps + 1):
            X_seq.append(X[i:i + lookback])
            y_seq.append(y[i + lookback:i + lookback + output_steps])
        return np.array(X_seq), np.array(y_seq), scalers  # dummy var so the func signature stays the same
    else:
        X_seq, y_seq, scalers = [], [], []

        for i in range(len(X) - lookback - output_steps + 1):
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_window = X[i:i + lookback]
            y_window = y[i + lookback:i + lookback + output_steps]

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
            # X = scaler_X.fit_transform(X.reshape(-1, 1))
            y = scaler_y.fit_transform(y.reshape(-1, 1))

        for output_step in output_steps:
            print("output_step=", output_step)

            X_multi, Y_multi, scalers = create_sequences(X, y, lookback, output_step)
            Y_multi = Y_multi.reshape(Y_multi.shape[0], Y_multi.shape[1])
            n_features = X.shape[1]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_multi, Y_multi, test_size=0.2, shuffle=False)

            if instance_norm:
                scalers_test = scalers[-len(y_test):]

            with tf.device(f'/GPU:{GPU_ID}'):
                # Define the LSTM Network
                model = tf.keras.Sequential([
                    # Input shape should be (lookback, features)
                    tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                                         return_sequences=True, input_shape=(lookback, n_features)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(output_step)  # Output layer for multi-step predictions
                ])

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                              loss='mse', metrics=['mae'])

                # Train the Model
                history = model.fit(
                    X_train, y_train.reshape(y_train.shape[0], y_train.shape[1]),
                    validation_split=0.25, epochs=100, batch_size=16, verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True
                        ),
                    ]
                )

                # Make Predictions
                y_pred_scaled = model.predict(X_test)

            # inverse scaling
            if instance_norm:
                y_pred_inverse = np.zeros_like(y_pred_scaled)
                y_test_inverse = np.zeros_like(y_test)
                for i, pred in enumerate(y_pred_scaled):
                    y_pred_inverse[i] = scalers_test[i].inverse_transform(pred.reshape(1, -1)).reshape(-1)
                    y_test_inverse[i] = scalers_test[i].inverse_transform(y_test[i].reshape(1, -1)).reshape(-1)
            else:
                y_pred_inverse = scaler_y.inverse_transform(y_pred_scaled)
                y_test_inverse = scaler_y.inverse_transform(y_test)

            # Calculate MAPE
            mape = np.mean(np.abs((y_pred_inverse - y_test_inverse) / y_test_inverse)) * 100
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            mapes.append(mape)

            # Plot the Full Price Data and Predictions
            plt.figure(figsize=(15, 8))

            # Plot Historical Price Data
            plt.plot(data_df.index, data_df['Price'], label="Historical Prices", color="blue", linewidth=2)

            # Add Test Predictions to the Plot
            start_test_index = len(data_df) - len(y_test)
            for i, prediction in enumerate(y_pred_inverse):
                forecast_dates = [data_df.index[start_test_index + i] + pd.DateOffset(days=j) for j in
                                  range(output_step)]
                plt.plot(forecast_dates, prediction, color="red", marker="o", linestyle="-",
                         label="Predicted" if i == 0 else "")

            plt.title("Carbon Credit Price Multi-Step Forecasting with LSTM")
            plt.xlabel("Time")
            plt.ylabel("Carbon Price")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.grid(True)
            plt.tight_layout()

            # Save Plot
            plot_file = os.path.join(result_folder, f"LSTM_MultiStep_feat{TopFeaturePerc}_lb{lookback}_pl{output_step}.png")
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
            results_file = os.path.join(result_folder, f"LSTM_MultiStep_feat{TopFeaturePerc}_lb{lookback}_pl{output_step}.xlsx")
            results_df.to_excel(results_file, index=False, engine="openpyxl")
            print(f"Results saved to {results_file}")

        mapes_dfs["MAPE_feat"+str(TopFeaturePerc)+"_train"+str(train_time)] = mapes

mapes_dfs.to_csv(os.path.join(result_folder,"mapes_lookbacks_multiSteps.csv"))
# mapes_steps = mapes_dfs
print(mapes_dfs)


