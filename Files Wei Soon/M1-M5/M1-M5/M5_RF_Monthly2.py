import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Paths
data_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/"
feature_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/features_monthly/"
target_file = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/FilteredData/target_monthly/1-historical_price-Monthly.xlsx"
result_folder = "C:/Users/jiangl2/PycharmProjects/carbonpriceforecast/data/InfluencingFactors/results/compare_models_monthly"
TopFeaturePerc = 0.2 # Select top 40% of the full set of features
# Create Features and Multi-Step Targets
forecast_horizon = 4  # Steps ahead to forecast
lookback = 10  # Number of past months used for prediction

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
sel_features_df0 = pd.read_excel(os.path.join(result_folder, "ranked_features_monthly.xlsx"))
sel_feature_len = int(TopFeaturePerc*len(sel_features_df0))
sel_features_df0.sort_values(by="Correlation", ascending=False)
sel_feature_names = sel_features_df0["Factor"][0:sel_feature_len]
print("sel_feature_names=",sel_feature_names)
print("sel_feature_length=",sel_feature_len)

data = data_df[sel_feature_names]
data.insert(0,'Price',data_df['Price'])
# Separate Input Features and Target
X = data.iloc[:, 1:].values  # All columns except the first (target price)
y = data['Price'].values  # The target price

# # Normalize Features and Target
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
#
# X_normalized = scaler_X.fit_transform(X)
# y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Scale data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
# X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

def create_features_targets(data, target, lookback, forecast_horizon):
    X, Y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i + lookback].flatten())  # Flatten features
        Y.append(target[i + lookback:i + lookback + forecast_horizon])
    return np.array(X), np.array(Y)

# y_multi = np.array([y[i:i + forecast_horizon] for i in range(len(y) - forecast_horizon)])
X_multi, Y_multi = create_features_targets(X_scaled, y_scaled, lookback, forecast_horizon)
Y_multi = Y_multi.reshape(Y_multi.shape[0], Y_multi.shape[1])
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_multi, Y_multi, test_size=0.2, shuffle=False)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_normalized = rf_model.predict(X_test)

# Inverse Transform Predictions and True Values
y_test_inverse = scaler_y.inverse_transform(y_test)
y_pred_inverse = scaler_y.inverse_transform(y_pred_normalized)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

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
plot_file = os.path.join(result_folder, "RF_MultiStep.png")
plt.savefig(plot_file, dpi=300)
print(f"Plot saved to {plot_file}")

plt.show()

# Save Results to Excel
results_df = pd.DataFrame()
for step in range(forecast_horizon):
    results_df[f"True_Step_{step + 1}"] = y_test_inverse[:, step]
    results_df[f"Predicted_Step_{step + 1}"] = y_pred_inverse[:, step]
results_file = os.path.join(result_folder, "RF_MultiStep.xlsx")
results_df.to_excel(results_file, index=False, engine="openpyxl")
print(f"Results saved to {results_file}")
