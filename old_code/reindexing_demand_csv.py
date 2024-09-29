import pandas as pd

df = pd.read_csv('data/demand_data_all_cleaned.csv', index_col=0, parse_dates=True)

print(df.info())
print('Has nulls:\n', df[df.isna().any(axis=1)])
print(pd.infer_freq(df.index))

# df = df.asfreq(freq='h')
#
# print(df.index.freq)
# print('Has nulls:\n', df[df.isna().any(axis=1)])
#
# def get_period(hour):
#     if 0 <= hour < 6:
#         return 'Night'
#     elif 6 <= hour < 12:
#         return 'Morning'
#     elif 12 <= hour < 18:
#         return 'Afternoon'
#     else:
#         return 'Evening'
#
# df['year'] = df.index.year
# df['month'] = df.index.month
# df['day'] = df.index.day
# df['hour'] = df.index.hour
# df['period'] = df['hour'].apply(get_period)
# df['day_of_week'] = df.index.strftime('%A')  # Full name of the day
# df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(bool)
#
# df['weatherDesc'] = df['weatherDesc'].ffill()
#
# float_columns = list(df.columns[:4]) + list(df.columns[11:18]) + list(df.columns[19:]) # Add all your float column names
# df[float_columns] = df[float_columns].interpolate(method='time')
#
# print(df.info())
# print('Has nulls:\n', df[df.isna().any(axis=1)])
# print()
# print(df)
# df.to_csv('data/demand_data_all_cleaned.csv')
#
# df['period'] = pd.Categorical(df['period']).codes
# df['day_of_week'] = pd.Categorical(df['day_of_week']).codes
# df['is_weekend'] = pd.Categorical(df['is_weekend']).codes
# df['weatherDesc'] = pd.Categorical(df['weatherDesc']).codes
#
# df.to_csv('data/demand_data_all_cleaned_numerical.csv')
