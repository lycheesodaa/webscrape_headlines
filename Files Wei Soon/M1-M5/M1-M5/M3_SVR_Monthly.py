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
feature_folder = "/home/lianlian/models/Time-LLM/dataset/carbon/features_monthly/"  # Folder containing your 24 Excel files
target_file = "/home/lianlian/models/Time-LLM/dataset/carbon/target_monthly/1-historical_price-Monthly.xlsx"  # File with carbon credit price data
result_folder = "/home/lianlian/webscrape_headlines/results/compare_models_monthly/SVR"
output_steps = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # Predict X steps ahead
lookbacks = [2, 8, 14, 20] # Number of past months to use as input
train_times = 1
fea_select_flag = 1 # choose with or without feature selection
TopFeaturePerc = 0 # Select top 40% of the full set of features
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


# Load and preprocess the data
data0 = load_and_merge_features(feature_folder, target_file)
data0.set_index('Month-Year', inplace=True)  # Use 'Month-Year' as the index
data_df = pd.DataFrame(data0)
# data_df.to_csv(os.path.join(result_folder,'merged_data.csv'))

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

# Prepare LSTM input
def create_sequences(X, y, lookback, output_steps):
    if not instance_norm:
        X_seq, y_seq, scalers = [], [], []
        for i in range(len(X) - lookback - output_steps + 1):
            X_seq.append(X[i:i + lookback])
            y_seq.append(y[i + lookback:i + lookback + output_steps])
        return np.array(X_seq), np.array(y_seq), scalers # dummy var so the func signature stays the same
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
for output_step in output_steps:
    print("output_step=", output_step)

    for train_time in range(0,train_times):
        print('train time=', train_time)
        mapes = []

        for i_lookback, lookback in enumerate(lookbacks):

            X_seq, y_seq, scalers = create_sequences(X, y, lookback, output_step)

            # Split into training and testing sets
            train_size = int(len(X_seq) * 0.6)
            test_size = int(len(X_seq) * 0.2)
            X_train,X_valid, X_test = X_seq[:train_size],X_seq[train_size:train_size+test_size], X_seq[train_size+test_size:]
            y_train,y_valid, y_test = y_seq[:train_size],y_seq[train_size:train_size+test_size], y_seq[train_size+test_size:]

            if instance_norm:
                scalers_test = scalers[-len(y_test):]

            # Multi-output SVR
            multi_svr = MultiOutputRegressor(SVR(kernel='rbf', C=1000, epsilon=0.01))
            X_train = X_train.reshape(X_train.shape[0], -1)
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
            multi_svr.fit(X_train, y_train)  # Train on all steps simultaneously

            # Calculate Losses
            training_loss = []
            validation_loss = []

            for i in range(y_train.shape[1]):  # Iterate through each output step
                y_train_pred = multi_svr.predict(X_train.reshape(X_train.shape[0], -1))[:, i]
                y_valid_pred = multi_svr.predict(X_valid.reshape(X_valid.shape[0], -1))[:, i]

                train_loss = mean_squared_error(y_train[:, i], y_train_pred)
                valid_loss = mean_squared_error(y_valid[:, i], y_valid_pred)

                training_loss.append(train_loss)
                validation_loss.append(valid_loss)

            # Plot Losses
            # plt.figure(figsize=(10, 6))
            # plt.plot(range(1, len(training_loss) + 1), training_loss, label="Training Loss", marker="o")
            # plt.plot(range(1, len(validation_loss) + 1), validation_loss, label="Validation Loss", marker="x")
            # plt.xlabel("Output Steps")
            # plt.ylabel("Mean Squared Error (MSE)")
            # plt.title("Training and Validation Loss per Output Step")
            # plt.legend()
            # plt.grid()
            # plt.show()
            # plt.savefig(os.path.join(result_folder, "CPF_Loss_Stepahead" + str(output_step) + "_Time" + str(train_time) + "_Lookback" + str(i_lookback) + ".png"), dpi=300, bbox_inches='tight')
            # plt.close()

            y_pred_scaled = multi_svr.predict(X_test.reshape(X_test.shape[0], -1))
            y_test = y_test.reshape(y_test.shape[0], -1)

            # Inverse scale the predictions and ground truth
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
                forecast_dates = [data_df.index[start_test_index + i] + pd.DateOffset(months=j) for j in
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
            plot_file = os.path.join(result_folder, f"SVR_MultiStep_lb{lookback}_pl{output_step}.png")
            plt.savefig(plot_file, dpi=300)
            print(f"Plot saved to {plot_file}")
            # plt.show()
            plt.close()

            # Save Results to Excel
            results_df = pd.DataFrame()
            for step in range(output_step):
                results_df[f"True_Step_{step + 1}"] = y_test_inverse[:, step]
                results_df[f"Predicted_Step_{step + 1}"] = y_pred_inverse[:, step]
            results_file = os.path.join(result_folder, f"SVR_MultiStep_lb{lookback}_pl{output_step}.xlsx")
            results_df.to_excel(results_file, index=False, engine="openpyxl")
            print(f"Results saved to {results_file}")

        mapes_dfs["MAPE_step"+str(output_step)+"_train"+str(train_time)]= mapes

mapes_dfs.to_csv(os.path.join(result_folder,"mapes_lookbacks_multiSteps.csv"))
# mapes_steps = mapes_dfs
print(mapes_dfs)


