# Feature selection using RandomForest
def feature_selection(X, y, feature_names):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    selector = SelectFromModel(model, prefit=True, threshold="3*mean")
    selected_features = selector.transform(X)
    selected_feature_names = feature_names[selector.get_support()]
    print(f"Selected Features: {list(selected_feature_names)}")
    print(f"Number of Selected Features: {len(list(selected_feature_names))}")
    pd.DataFrame(selected_feature_names).to_excel(os.path.join(result_folder,"selected_feature_names.xlsx"))
    return selected_features, selected_feature_names
    


# Load and preprocess the data
data = load_and_merge_features(feature_folder, target_file)
data.set_index('Month-Year', inplace=True)  # Use 'Month-Year' as the index
data_df = pd.DataFrame(data)
data_df.to_csv(os.path.join(result_folder,'merged_data.csv'))

# Include historical prices as features (lagged values)
historical_lags = 1  # Number of lagged months to include
for lag in range(1, historical_lags + 1):
    data[f"Historical Price (Lag {lag})"] = data.iloc[:, 0].shift(lag)

# Drop rows with NaN values caused by lagging
data = data.dropna()

# Separate input features and target
X = data.iloc[:, 1:].values  # All columns except the first (target price)
y = data.iloc[:, 0].values  # First column is the target price

X_selected, selected_feature_names = feature_selection(X, y, feature_names)