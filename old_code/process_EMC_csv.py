import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime, timedelta


def period_to_time(period):
    # Convert period to minutes past midnight
    minutes = (int(period) - 1) * 30
    # Create a time object
    return (datetime.min + timedelta(minutes=minutes)).time()


dir_path = ("C:/Users/stucws/Documents/astar/data/demand_datasets/EMC dataset/"
            "uniform_singapore_energy_price_n_demand_forecast/")
file_list = os.listdir(dir_path)

all_dfs = []

for file in tqdm(file_list):
    df = pd.read_csv(dir_path + file)

    df.drop(columns=['INFORMATION TYPE'], inplace=True)

    df['DATE'] = pd.to_datetime(df['DATE'])

    # Create the time based on PERIOD
    df['TIME'] = df['PERIOD'].apply(period_to_time)

    # Combine DATE and TIME to create a full timestamp
    df['TIMESTAMP'] = df.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)

    # Set the new column as the index
    df.set_index('TIMESTAMP', inplace=True)

    df.drop(columns=['DATE', 'PERIOD', 'TIME'])

    df.sort_index(inplace=True)

    all_dfs.append(df)

final_result = pd.concat(all_dfs)

final_result.sort_index(inplace=True)

final_result.to_csv('EMC_Energy_Price_and_Demand.csv')