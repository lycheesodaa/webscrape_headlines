import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import calendar


def get_last_day_of_month(year_, month_):
    _, last_day_ = calendar.monthrange(year_, month_)
    return date(year_, month_, last_day_)


time_now = datetime.now()
start_date = date(2014, 1, 1)
end_date = date(2024, 6, 1)
current_date = start_date

all_data = []
count = 0

while current_date <= end_date:
    year = current_date.year
    month = current_date.month
    last_day = get_last_day_of_month(year, month)
    print(f"{current_date.strftime('%B %Y')}: Last day is {last_day.strftime('%Y-%m-%d')}")

    url = (f"https://api.worldweatheronline.com/premium/v1/past-weather.ashx"
           f"?q=Singapore&date={current_date.strftime('%Y-%m-%d')}&enddate={last_day.strftime('%Y-%m-%d')}"
           f"&tp=1&format=json&key=fbc57f8ee9d4461e8e524518242507")

    current_date += relativedelta(months=1)

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    try:
        data = response.json()['data']
    except Exception as e:
        print(e)
        exit()

    parsed_data = []
    daily_data = data['weather']

    for daily in daily_data:
        actual_date = daily['date']
        hourly_data = daily['hourly']

        # Parse each hourly entry
        for entry in hourly_data:
            hour = int(entry['time']) // 100  # Convert '100' to 1, '200' to 2, etc.
            dt = datetime.strptime(f"{actual_date} {hour:02d}:00", "%Y-%m-%d %H:%M")

            parsed_data.append({
                'datetime': dt,
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'period': 'Night' if 0 <= dt.hour < 6 else 'Morning' if 6 <= dt.hour < 12 else 'Afternoon' if 12 <= dt.hour < 18 else 'Evening',
                'day_of_week': dt.strftime('%A'),
                'is_weekend': dt.weekday() >= 5,
                'humidity': int(entry['humidity']),
                'tempC': float(entry['tempC']),
                'heatIndexC': float(entry['HeatIndexC']),
                'precipMM': float(entry['precipMM']),
                'windspeedKmph': float(entry['windspeedKmph']),
                'winddirDegree': float(entry['winddirDegree']),
                'windGustKmph': float(entry['WindGustKmph']),
                'weatherDesc': str(entry['weatherDesc']),
                'visibility': float(entry['visibility']),
                'pressure': float(entry['pressure']),
                'cloudcover': float(entry['cloudcover']),
                'dewPointC': float(entry['DewPointC']),
                'uvIndex': float(entry['uvIndex']),
                'feelsLikeC': float(entry['FeelsLikeC']),
            })

    all_data.extend(parsed_data)
    count += 1

    # if count % 10 == 0:
    #     df = pd.DataFrame(all_data)
    #     df.set_index('datetime', inplace=True)
    #     df.sort_index(inplace=True)
    #     df = df[~df.index.duplicated(keep='first')]
    #     df.to_csv('sg_historical_hourly_weather.csv')


df = pd.DataFrame(all_data)
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)
df = df[~df.index.duplicated(keep='first')]
df.to_csv('sg_historical_hourly_weather.csv')
