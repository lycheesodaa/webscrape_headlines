import os
import datetime as dt
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import json
from decouple import config
from langdetect import detect
import time
import sqlite3
import bs4 as bs
import pandas as pd
import yfinance as yf
import ta


# def get_snp_symbols():
#     """Method that gets the stock symbols from companies listed in the S&P 500 from Wikipedia
#
#     Return
#     ------
#     `tickers` : list
#         S&P 500 company symbols
#     """
#     resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
#     soup = bs.BeautifulSoup(resp.text, 'lxml')
#     table = soup.find_all('table')[0]  # Grab the first table
#
#     symbols = []
#     for row in table.findAll('tr')[1:]:
#         ticker = row.findAll('td')[0].text.strip('\n')
#         symbols.append(ticker)
#     print(f"DEBUG: Total number of stock symbols: {len(symbols)}")
#
#     return symbols
#
#
# def get_exclusions():
#     # file_list = os.listdir("yfinance_ticker_data")
#     file_list = os.listdir("../yfinance_ticker_data/metadata")
#     file_names_without_extension = [os.path.splitext(f)[0] for f in file_list]
#
#     return file_names_without_extension
#
#
# tickers = get_snp_symbols()
# exclusions = get_exclusions()
# tickers = [symbol.replace('.', '-') for symbol in tickers if symbol not in exclusions]
#
# for i, ticker in enumerate(tickers):
#     print(f"DEBUG: Getting {ticker} ({i}/{len(tickers)}) data...")
#     stock = yf.Ticker(ticker)
#     df = stock.history(
#         period='max',
#         interval='1d',
#         start='2000-01-01',
#         end=dt.datetime.now(),
#         prepost=True,
#         actions=True,
#         raise_errors=True
#     )
#     # df.to_csv(f'yfinance_ticker_data/{ticker}.csv')
#     json.dump(stock.info, open(f'yfinance_ticker_data/metadata/{ticker}.json', 'w'))

directory = "external_data/FNSPID/full_history/"

file_list = os.listdir(directory)
for csv_file in file_list:
    if csv_file.endswith('.csv'):
        print(f'Processing {csv_file}...')
        df = pd.read_csv(directory + csv_file)
        df = ta.add_all_ta_features(
            df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True
        )
        print(f'Saving {csv_file}...')
        df.to_csv(directory + csv_file, index=False)