import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
import os
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot as plt


# Paths
data_folder = "C:/Users/stucws/Documents/astar/data/carbon/"
feature_folder = "C:/Users/stucws/Documents/astar/data/carbon/features_monthly/"  # Folder containing your 24 Excel files
target_file = "C:/Users/stucws/Documents/astar/data/carbon/target_monthly/1-historical_price-Monthly.xlsx"  # File with carbon credit price data
result_folder = "C:/Users/stucws/Documents/astar/sentiment-analysis/webscrape_headlines/results/carbon_forecasting/"
output_steps = [k for k in range(1, 10)]  # Number of months to forecast ahead
lookbacks = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]  # Number of past months to use as input
train_times = 1
fea_select_flag = 1  # choose with or without feature selection

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
data = load_and_merge_features(feature_folder, target_file)
data.set_index('Month-Year', inplace=True)  # Use 'Month-Year' as the index
data_df = pd.DataFrame(data)
data_df.to_csv(os.path.join(result_folder,'merged_data.csv'))

# Include historical prices as features (lagged values)
historical_lags = 1  # Number of lagged months to include
for lag in range(1, historical_lags + 1):
    data_df[f"Historical Price (Lag {lag})"] = data_df.iloc[:, 0].shift(lag)
# Drop rows with NaN values caused by lagging
data_df = data_df.dropna()
comb_df = data_df

# Correlation analysis
corr_factors = [col for col in comb_df.columns if col != 'Price']
corr_values = [comb_df[col].corr(comb_df['Price'], method='spearman') for col in corr_factors]

# Create a DataFrame for correlation values
corr_matrix_df1 = pd.DataFrame({"Factor": corr_factors, "Correlation": corr_values})
corr_matrix_df2 = pd.DataFrame({"Factor": corr_factors, "Correlation": [abs(value) for value in corr_values]})
# Save to CSV
corr_matrix_df1.to_csv(os.path.join(result_folder, "corr_matrix.csv"), index=False)

# Sort by correlation in descending order
sorted_corr_df = corr_matrix_df1.sort_values(by="Correlation", ascending=False)

# Sort by the absolute value of the correlation in descending order
abs_sorted_corr_df = corr_matrix_df2.sort_values(by="Correlation", ascending=True)
# abs_sorted_corr_df[-int(len(abs_sorted_corr_df) * 0.2):].T.to_csv(
#     os.path.join(result_folder, 'top_20pct_feats.csv'), index=False, header=False)

# Calculate dynamic figure size based on the number of factors
num_factors = len(sorted_corr_df)
fig_height = min(10, max(4, num_factors * 0.5))  # Scale between 4 and 10 inches

# Plot horizontal bar chart-------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(22, fig_height))
ax.barh(sorted_corr_df["Factor"], sorted_corr_df["Correlation"], color="skyblue", edgecolor="black")
# Add labels and title
ax.set_xlabel("Correlation Factor", fontsize=12)
ax.set_ylabel("Factors", fontsize=12)
ax.set_title("Rank of Correlation Between Factors and Carbon Credit Price", fontsize=14)
# Adjust layout to center the plot and make space for labels
plt.subplots_adjust(left=0.4, right=0.95, top=0.5, bottom=0.1)
# Add grid and fine-tune spacing
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.tick_params(axis='y', labelsize=8)
# Tighten layout and display the plot
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(result_folder, "Correlation Ranking.png"))

# Plot horizontal bar chart absolute values--------------------------------------------------------------
fig, ax = plt.subplots(figsize=(22, fig_height))
ax.barh(abs_sorted_corr_df["Factor"], abs_sorted_corr_df["Correlation"], color="skyblue", edgecolor="black")
# Add labels and title
ax.set_xlabel("Correlation Factor", fontsize=12)
ax.set_ylabel("Factors", fontsize=12)
ax.set_title("Rank of Absolute Correlation Between Factors and Carbon Credit Price", fontsize=14)
# Adjust layout to center the plot and make space for labels
plt.subplots_adjust(left=0.4, right=0.95, top=0.5, bottom=0.1)
# Add grid and fine-tune spacing
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.tick_params(axis='y', labelsize=8)
# Tighten layout and display the plot
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(result_folder, "Absolute Correlation Ranking.png"))

abs_sorted_corr_df.to_excel(os.path.join(result_folder, "ranked_abs_features_monthly.xlsx"))
sorted_corr_df.to_excel(os.path.join(result_folder, "ranked_features_monthly.xlsx"))

