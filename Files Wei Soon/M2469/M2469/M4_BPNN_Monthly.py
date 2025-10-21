import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# Paths
data_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/"
feature_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/features_monthly/"  # Folder containing your 24 Excel files
target_file = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/target_monthly/1-historical_price-Monthly.xlsx"  # File with carbon credit price data
result_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/results/compare_models_monthly/BPNN"
output_steps = [3] # Number of months to forecast ahead
lookbacks = [12] # Number of past months to use as input
train_times = 1
fea_select_flag = 1 # choose with or without feature selection
TopFeaturePerc = 0.20 # Select top 40% of the full set of features

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
data_df.to_csv(os.path.join(result_folder,'merged_data.csv'))

# Include historical prices as features (lagged values)
historical_lags = 1  # Number of lagged months to include
for lag in range(1, historical_lags + 1):
    data_df[f"Historical Price (Lag {lag})"] = data_df.iloc[:, 0].shift(lag)
data_df = data_df.dropna()

# Load in selected features based on the spearman correlation analysis
sel_features_df0 = pd.read_excel(os.path.join(result_folder, "ranked_features_monthly.xlsx"))
sel_feature_len = int(TopFeaturePerc*len(sel_features_df0))
sel_features_df0.sort_values(by="Correlation", ascending=False)
sel_feature_names = sel_features_df0["Factor"][0:sel_feature_len]
print("sel_feature_names=",sel_feature_names)
print("sel_feature_length=",sel_feature_len)

data = data_df[sel_feature_names]
data.insert(0,'Price',data_df['Price'])

# Separate input features and target
X = data.iloc[:, 1:].values  # All columns except the first (target price)
y = data.iloc[:, 0].values  # First column is the target price

# Scale data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
# X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Prepare LSTM input
def create_sequences(X, y, lookback, output_steps):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback - output_steps + 1):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback:i + lookback + output_steps])
    return np.array(X_seq), np.array(y_seq)

mapes_dfs = pd.DataFrame()
for output_step in output_steps:
    print("output_step=", output_step)

    for train_time in range(0,train_times):
        print('train time=', train_time)
        mapes = []

        for i_lookback, lookback in enumerate(lookbacks):

            X_multi, Y_multi = create_sequences(X_scaled, y_scaled, lookback, output_step)
            X_multi_reshaped = X_multi.reshape(X_multi.shape[0], -1)
            Y_multi = Y_multi.reshape(Y_multi.shape[0], Y_multi.shape[1])
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_multi_reshaped, Y_multi, test_size=0.2, shuffle=False)

            # Define the Multi-Layer Neural Network
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_step)  # Output layer for multi-step predictions
            ])
            # Compile the Model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            # Train the Model
            history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1]), validation_split=0.2,
                                epochs=100, batch_size=32, verbose=1)

            # Make Predictions
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_inverse = scaler_y.inverse_transform(y_test.reshape(y_test.shape[0], -1))

            # Calculate MAPE
            mape = np.mean(np.abs((y_pred - y_test_inverse) / y_test_inverse)) * 100
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            mapes.append(mape)

            # Save predictions
            prediction_df = pd.DataFrame({"Month-Year": data.index[-len(y_test):]})
            prediction_df_p = pd.DataFrame({"Month-Year": data.index[-len(y_test):]})
            for step_ahead in range(0,output_step):
                prediction_df["Original Price (Step "+str(step_ahead)+")"] = y_test_inverse[:, step_ahead]
                prediction_df["Predicted Price (Step " + str(step_ahead)+")"] = y_pred[:, step_ahead]
            for step_ahead in range(0,output_step):
                  prediction_df_p["Predicted Price (Step " + str(step_ahead)+")"] = y_pred[:, step_ahead]

            prediction_df.to_excel(os.path.join(result_folder,"CPF_Stepahead"+str(output_step)+"_Time"+str(train_time)+"_Lookback"+str(i_lookback)+".xlsx"), index=False)

            # Plotting
            plt.figure(figsize=(10, 6))
            steps = range(len(prediction_df["Original Price (Step 0)"]))
            plt.plot(steps, prediction_df["Original Price (Step 0)"], label="Actual", color="orange", linewidth=2)
            for i, row in enumerate(prediction_df_p.values):
                # x = range(1, len(row) + 1)# Dynamic x-axis based on the number of columns
                x = [ii for ii in range(i, i + prediction_df_p.values.shape[1]-1)]
                plt.plot(x, row[1:], color="blue", marker="o", linestyle="-", markersize=5, label="Pred" if i == 0 else "")
            plt.savefig(os.path.join(result_folder, "CPF_Stepahead"+str(output_step)+"_Time"+str(train_time)+"_Lookback"+str(i_lookback)+".png"), dpi=300, bbox_inches='tight')
            plt.close()

        mapes_dfs["MAPE_step"+str(output_step)+"_train"+str(train_time)]= mapes

mapes_dfs.to_csv(os.path.join(result_folder,"mapes_lookbacks_multiSteps.csv"))
# mapes_steps = mapes_dfs
print(mapes_dfs)


