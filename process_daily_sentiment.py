import os

import pandas as pd
from datetime import datetime, timedelta

directory = 'external_data/FNSPID/top50_news_w_content/with_emotion_content/'
output_dir = 'external_data/FNSPID/candle_w_sentiment/day_average_content/'
candle_dir = 'external_data/FNSPID/full_history/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

files = os.listdir(directory)

for file in files:
    # Read the CSV file
    df = pd.read_csv(directory + file)

    # print(df.columns)
    cols = ['Date',
            'neutral', 'positive', 'negative',
            'sadness', 'neutral_emotion', 'fear', 'anger', 'disgust', 'surprise', 'joy']

    df = df[cols]

    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={'Date': 'date'}, inplace=True)

    # Set timestamp as index
    df.set_index('date', inplace=True)

    # Resample to daily frequency and calculate mean
    daily_avg = df.resample('B').mean()

    # Interpolate values for all sentiments
    daily_avg = daily_avg.interpolate()

    # Reset index to make date a column
    daily_avg.reset_index(inplace=True)
    daily_avg['date'] = pd.to_datetime(daily_avg['date'])
    

    # print('% missing:', len(daily_avg[daily_avg['neutral'].isna()]) / len(daily_avg) * 100)

    try:
        candlesticks = pd.read_csv(candle_dir + file)
    except FileNotFoundError:
        candlesticks = pd.read_csv(candle_dir + file.lower())

    candlesticks['date'] = pd.to_datetime(candlesticks['date'], utc=True)
    candlesticks.set_index('date', inplace=True)

    if pd.infer_freq(candlesticks.index) is None:
        print('Candlestick frequency not inferrable. Casting to B freq.')
        candlesticks = candlesticks.resample('B')

    combined = pd.merge(candlesticks, daily_avg, how='inner', on='date')
    combined['date'] = pd.to_datetime(combined['date'])
    combined.set_index('date', inplace=True)

    # Save to CSV
    combined.to_csv(output_dir + file)

    print(f"Daily averaged sentiment scores have been saved to {output_dir + file}")