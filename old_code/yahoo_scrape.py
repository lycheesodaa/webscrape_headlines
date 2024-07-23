import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import bs4 as bs
import pandas as pd
import time
import random as rd


# time.sleep(rd.randint(1, 5))

def get_snp_symbols():
    """Method that gets the stock symbols from companies listed in the S&P 500 from Wikipedia

    Return
    ------
    `tickers` : list
        S&P 500 company symbols
    """
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find_all('table')[0]  # Grab the first table

    symbols = {}
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip('\n')
        security = row.findAll('td')[1].text.strip('\n')
        symbols[ticker] = security
    print(f"DEBUG: Total number of stock symbols: {len(symbols)}")
    print(symbols['AAPL'])

    return symbols


def get_exclusions():
    file_list = os.listdir("../yfinance_ticker_data/news")
    file_names_without_extension = [os.path.splitext(f)[0] for f in file_list]

    return file_names_without_extension


def iterate_pages(ticker, security):
    url = 'https://finance.yahoo.com/xhr/ncp'

    params = {
        'location': 'US',
        'queryRef': 'newsAll',
        'serviceKey': 'ncp_fin',
        'listName': f'{ticker}-news-related',
        'lang': 'en-US',
        'region': 'US'
    }

    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'text/plain;charset=UTF-8',
        'cookie': 'A1=d=AQABBKe9c2YCEHgOQa-iAicF3onWY0cw6EsFEgEBAQEPdWZ9Zq9D8HgB_eMAAA&S=AQAAApQCFsTKbs9-TsrYcv6_tW0; A3=d=AQABBKe9c2YCEHgOQa-iAicF3onWY0cw6EsFEgEBAQEPdWZ9Zq9D8HgB_eMAAA&S=AQAAApQCFsTKbs9-TsrYcv6_tW0; axids=gam=y-sUsU0pZE2uJLMwWKVrbGHnrAOIhhQqig~A&dv360=eS1KTzJiOHVSRTJ1RVB6aS5tMTJPNU1Ib2JXNTFKc19Zan5B&ydsp=y-H4E38mVE2uJtopTX65a7KN95fmdS1nAh~A&tbla=y-nWxyyPhE2uJ98Y1tQrMtOZNX17fYY54Q~A; tbla_id=5b1e5b9c-956d-4e5f-bbfe-3a866904dfcd-tuctd6d432e; trc_cookie_storage=taboola%2520global%253Auser-id%3D5b1e5b9c-956d-4e5f-bbfe-3a866904dfcd-tuctd6d432e; gpp=DBAA; gpp_sid=-1; A1S=d=AQABBKe9c2YCEHgOQa-iAicF3onWY0cw6EsFEgEBAQEPdWZ9Zq9D8HgB_eMAAA&S=AQAAApQCFsTKbs9-TsrYcv6_tW0; cmp=t=1719282089&j=0&u=1---; PRF=t%3DAAPL%252BNVDA%252B9988.HK%252BBRK-B%26qct-neo%3Dcandle',
        'origin': 'https://finance.yahoo.com',
        'priority': 'u=1, i',
        'referer': 'https://finance.yahoo.com/quote/AAPL/news/',
        'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
    }

    # fencepost for the initial dataframe
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    df = pd.DataFrame.from_dict(data['results'])
    time.sleep(rd.randint(1, 5))
    print('Total news entries: {}'.format(data['total']))

    page = 2  # page number starts at 1 and is implicit for the fencepost

    # iterate until 'results' segment is empty
    while True:
        params['page'] = page

        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if 'results' not in data or len(data['results']) == 0:
            break
        df = pd.concat([df, pd.DataFrame.from_dict(data['results'])])

        page += 1
        time.sleep(rd.randint(1, 5))

    print(f'Total number of news articles pulled: {len(df)}')
    return df


def get_news():
    snp500 = get_snp_symbols()
    exclusions = get_exclusions()
    tickers = [symbol for symbol, security in snp500.items() if symbol not in exclusions]

    for i, ticker in enumerate(tickers):
        print(f'Getting {ticker} news data ({i}/{len(tickers)})...')
        ticker_df = iterate_pages(ticker, snp500[ticker])
        ticker_df.to_csv(f'yfinance_ticker_data/news/{ticker}.csv')
        print('Export done.')


get_news()
