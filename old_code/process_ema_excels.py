import pandas as pd
import os
from tqdm import tqdm

dir_path = "C:/Users/stucws/Documents/astar/data/demand_datasets/raw/EMA dataset/"
file_list = os.listdir(dir_path)
file_list = [item for item in file_list if not item.endswith(".zip")]

all_dfs = []

for file in tqdm(file_list):
    df = pd.read_excel(dir_path + file)

    # Extract the date row and create a list of dates
    dates = df.iloc[0, 1::3]

    # Extract the data rows
    data_rows = df.iloc[4:52]

    formatted_data = []

    for _, row in data_rows.iterrows():
        time = row.iloc[0]
        for i in range(7):  # 7 days in the data
            date = str(dates.iloc[i]).split()[0]
            system_demand = row.iloc[1 + i * 3]
            nem_demand_actual = row.iloc[2 + i * 3]
            nem_demand_forecast = row.iloc[3 + i * 3]

            formatted_data.append({
                'period_end_datetime': f"{date} {time}:00",
                'system_demand_actual': system_demand,
                'nem_demand_actual': nem_demand_actual,
                'nem_demand_forecast': nem_demand_forecast
            })

    result_df = pd.DataFrame(formatted_data)
    result_df['period_end_datetime'] = pd.to_datetime(result_df['period_end_datetime'])
    result_df = result_df.sort_values('period_end_datetime')

    all_dfs.append(result_df)

final_results = pd.concat(all_dfs, ignore_index=True)

final_results = final_results.sort_values('period_end_datetime')

final_results.to_csv('formatted_energy_demand_all.csv', index=False)