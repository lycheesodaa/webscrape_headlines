import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Paths
data_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/"
feature_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/features_monthly/"  # Folder containing your 24 Excel files
target_file = "/home/lianlian/models/Time-LLM/dataset/carbon/target_monthly/1-historical_price-Monthly.xlsx"  # File with carbon credit price data
result_folder = "/home/lianlian/webscrape_headlines/results/compare_models_monthly/CNN"
TopFeaturePerc = 0 # Select top 40% of the full set of features
# Create Features and Multi-Step Targets
forecast_horizons = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # Predict X steps ahead
lookbacks = [8, 14, 20] # Number of past months to use as input
GPU_ID = 0
instance_norm = True
if instance_norm:
    result_folder = Path(result_folder) / 'instance_norm'
    result_folder.mkdir(parents=True, exist_ok=True)


# Load all feature files and merge them
def load_and_merge_features(folder_path, target_file):
    # Load the target price data
    target_df = pd.read_excel(target_file, engine='openpyxl')
    target_df['Month-Year'] = pd.to_datetime(target_df['Month-Year'], format='%b-%Y')

    # Merge all features
    all_features = []
    feature_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # Function to extract the numerical prefix for sorting
    def extract_number(file_name):
        # Split the filename by the hyphen and return the numerical part as an integer
        return int(file_name.split('-')[0])
    # Sort the files based on the numerical prefix
    feature_files = sorted(feature_files, key=extract_number)

    for file in feature_files:
        feature_df = pd.read_excel(os.path.join(folder_path, file), engine='openpyxl')
        feature_df['Month-Year'] = pd.to_datetime(feature_df['Month-Year'], format='%b-%Y')
        all_features.append(feature_df)

    # Merge all features with the target price data
    merged_data = target_df
    for feature_df in all_features:
        merged_data = pd.merge(merged_data, feature_df, on='Month-Year', how='inner')
    return merged_data

# Load and Preprocess Data
data0 = load_and_merge_features(feature_folder, target_file)
data0.set_index('Month-Year', inplace=True)
data_df = pd.DataFrame(data0)

# Include historical prices as features (lagged values)
historical_lags = 1  # Number of lagged months to include
for lag in range(1, historical_lags + 1):
    data_df[f"Historical Price (Lag {lag})"] = data_df.iloc[:, 0].shift(lag)
data_df = data_df.dropna()

# Load in selected features based on the spearman correlation analysis
sel_features_df0 = pd.read_excel(os.path.join(data_folder, "res/ranked_abs_features_monthly.xlsx"))
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

def create_features_targets(X, y, lookback, forecast_horizon):
    if not instance_norm:
        X_seq, y_seq, scalers = [], [], []
        for i in range(len(X) - lookback - forecast_horizon + 1):
            X_seq.append(X[i:i + lookback])
            y_seq.append(y[i + lookback:i + lookback + forecast_horizon])
        return np.array(X_seq), np.array(y_seq), scalers # dummy var so the func signature stays the same
    else:
        X_seq, y_seq, scalers = [], [], []

        for i in range(len(X) - lookback - forecast_horizon + 1):
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_window = X[i:i + lookback]
            y_window = y[i + lookback:i + lookback + forecast_horizon]

            X_normalized = scaler_X.fit_transform(X_window)

            scaler_y.fit(X_window[:, :1])
            y_normalized = scaler_y.transform(y_window.reshape(-1, 1))

            X_seq.append(X_normalized)
            y_seq.append(y_normalized)
            scalers.append(scaler_y)

        return np.array(X_seq), np.array(y_seq), scalers


mapes_dfs = pd.DataFrame()
for forecast_horizon in forecast_horizons:
    print("forecast_horizon=", forecast_horizon)
    mapes = []

    for i_lookback, lookback in enumerate(lookbacks):
        print('lookback=', lookback)
        # y_multi = np.array([y[i:i + forecast_horizon] for i in range(len(y) - forecast_horizon)])
        X_multi, Y_multi, scalers = create_features_targets(X, y, lookback, forecast_horizon)
        Y_multi = Y_multi.reshape(Y_multi.shape[0], Y_multi.shape[1])
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_multi, Y_multi, test_size=0.2, shuffle=False)

        if instance_norm:
            scalers_test = scalers[-len(y_test):]

        with tf.device(f'/GPU:{GPU_ID}'):
            # Build CNN Model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(lookback, X_multi.shape[2])),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(forecast_horizon, activation='linear')  # Output layer predicts multiple steps
            ])

            # Compile the Model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

            # Train the Model
            history = model.fit(X_train, y_train, validation_split=0.25, epochs=100, batch_size=16, verbose=1,
                                callbacks=[tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    patience=10,
                                    restore_best_weights=True
                                ), ])

            # Make Predictions
            y_pred_scaled = model.predict(X_test)

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
        mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse) * 100
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        mapes.append(mape)

        # Plot the Full Price Data and Predictions
        plt.figure(figsize=(15, 8))

        # Plot Historical Price Data
        plt.plot(data_df.index, data_df['Price'], label="Historical Prices", color="blue", linewidth=2)

        # Add Test Predictions to the Plot
        start_test_index = len(data_df) - len(y_test)
        for i, prediction in enumerate(y_pred_inverse):
            forecast_dates = [data_df.index[start_test_index + i] + pd.DateOffset(months=j) for j in range(forecast_horizon)]
            plt.plot(forecast_dates, prediction, color="red", marker="o", linestyle="-", label="Predicted" if i == 0 else "")

        plt.title("Carbon Credit Price Multi-Step Forecasting with Random Forest")
        plt.xlabel("Time")
        plt.ylabel("Carbon Price")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()

        # Save Plot
        plot_file = os.path.join(result_folder, f"CNN_MultiStep_lb{lookback}_pl{forecast_horizon}.png")
        plt.savefig(plot_file, dpi=300)
        print(f"Plot saved to {plot_file}")
        # plt.show()
        plt.close()

        # Save Results to Excel
        results_df = pd.DataFrame()
        for step in range(forecast_horizon):
            results_df[f"True_Step_{step + 1}"] = y_test_inverse[:, step]
            results_df[f"Predicted_Step_{step + 1}"] = y_pred_inverse[:, step]
        results_file = os.path.join(result_folder, f"CNN_MultiStep_lb{lookback}_pl{forecast_horizon}.xlsx")
        results_df.to_excel(results_file, index=False, engine="openpyxl")
        print(f"Results saved to {results_file}")

    mapes_dfs["MAPE_step"+str(forecast_horizon)] = mapes

mapes_dfs.to_csv(os.path.join(result_folder,"mapes_lookbacks_multiSteps.csv"))
# mapes_steps = mapes_dfs
print(mapes_dfs)

