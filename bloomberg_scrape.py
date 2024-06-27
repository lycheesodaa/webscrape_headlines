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

    return symbols


def get_exclusions():
    file_list = os.listdir("ticker_data/news")
    file_names_without_extension = [os.path.splitext(f)[0] for f in file_list]

    return file_names_without_extension


def iterate_pages(ticker, security):
    url = 'https://www.bloomberg.com/markets2/api/search'
    params = {
        "query": security,
        "sort": "time:desc"
    }
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "cookie": "country_code=SG; seen_uk=1; _sp_krux=false; bbgconsentstring=req1fun1pad0; bdfpc=004.2737409235.1719222498332; AMP_MKTG_4c214fdb8b=JTdCJTdE; _parsely_session={\"sid\":1,\"surl\":\"https://www.bloomberg.com/view/topics/middle-east\",\"sref\":\"\",\"sts\":1719222498588,\"slts\":0}; _parsely_visitor={\"id\":\"pid=7ebc31d0-f5c5-4954-9da1-743ba6d99bdb\",\"session_count\":1,\"last_session_ts\":1719222498588}; consentUUID=faeee592-aade-4bb7-bb24-3a8d88a152f5; usnatUUID=b7900bcb-2935-4f2d-93c3-6ffad5548545; agent_id=e50b6510-b957-4c3e-960d-3d6375b2e530; session_id=8e029aa9-25b0-4f4b-95f3-0a878ea50938; _session_id_backup=8e029aa9-25b0-4f4b-95f3-0a878ea50938; session_key=8ed8b5c9985ccb236449b42a20859b8023921de2; gatehouse_id=a324a050-f2ea-4be5-9dc9-69e7e2d4d5a0; geo_info=%7B%22countryCode%22%3A%22SG%22%2C%22country%22%3A%22SG%22%2C%22field_n%22%3A%22cp%22%2C%22trackingRegion%22%3A%22Asia%22%2C%22cacheExpiredTime%22%3A1719827301874%2C%22region%22%3A%22Asia%22%2C%22fieldN%22%3A%22cp%22%7D%7C1719827301874; pxcts=e4af7659-320e-11ef-ada6-a5fc5728a0f6; _pxvid=e4af63f2-320e-11ef-ada5-6e8d8011b008; _ga=GA1.1.501270564.1719222499; geo_info={\"country\":\"SG\",\"region\":\"Asia\",\"fieldN\":\"cp\"}|1719827299277; _reg-csrf=s%3A7V5IPymG8uDm1Y93yZQyq6sO.Lc4GNB8bGwdW0lkXAc1kHcIP6yAt6vW4qQXxesUvR2k; _user-data=%7B%22status%22%3A%22anonymous%22%7D; exp_pref=AMER; __sppvid=9cbec9d3-c259-42b9-952e-cdac364aec72; _reg-csrf-token=y2YRUGWO-4t-Sj6Uqlq-12sfpZaBVXa2pBOg; __eoi=ID=d510cd0197d5f30f:T=1719222508:RT=1719223376:S=AA-AfjanZkBUd0m_ykifUuwGS126; _px3=471c14ff3fe713de9f550eff3c90c5b81766ad9bd1c0fc1bfaa291b01cbbf4c0:ZBc0hFKWeXf9h7ECOIwl1uy+F4mnr9H3GZFtWnooOoh7+kZ8D+XregoN3LUklrdwJf6Rw8vx92NEEJiuQjkPCQ==:1000:EkVgGwpdPr0C+pRE84h34Rnli1JavtVdrjNLQGpMm9Rv8bPsnLJZ7xUCXLYGYri16Zz5QTeB3WpoA7e9rTJx73T6ptDncsaX3sIvFEVUr3X89HOTtV8rLfHgVERKn5jw1ywwuvRgORrnu+qS5mJyhJrlLPw59hc7X8D3i0xdFxesXeS7ckrsG6CylNPcc+ULaEnglUFKp1e7X4jJqoz5aeudQBzwZoJiP+JDDgJUyYk=; _px2=eyJ1IjoiM2NlZmQxYjAtMzIxMC0xMWVmLThmNWUtYjNmNmExN2NlNDhjIiwidiI6ImU0YWY2M2YyLTMyMGUtMTFlZi1hZGE1LTZlOGQ4MDExYjAwOCIsInQiOjE3MTkyMjM5MDY3NzcsImgiOiIwNjY3NjMyYjg0NGY3NTRjYTA3YmM0MmZkZWIxY2JmMDU4YjBjYzMwODlmYzc1NWY2MzUwZjZhYjg1NDg4ZTRmIn0=; _pxde=c8e70f31e8a69ff3bd566ac220e7c83842f0b0025f4056496da0b5065501ae2a:eyJ0aW1lc3RhbXAiOjE3MTkyMjM2MDY3NzgsImZfa2IiOjAsImlwY19pZCI6W119; AMP_4c214fdb8b=JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjIwNWE2MzJhNS1iYjVkLTQ2ZTgtOTdmMS1kZDJmOWFmN2E5MTAlMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzE5MjIyNDk4NTU2JTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTcxOTIyMzYwNDQ5MyUyQyUyMmxhc3RFdmVudElkJTIyJTNBMzklMkMlMjJwYWdlQ291bnRlciUyMiUzQTclN0Q=; _ga_GQ1PBLXZCT=GS1.1.1719222498.1.1.1719223611.0.0.0",
        "if-none-match": 'W/"1772-cSjFitLqTmCpeDqF+yZGxF2wpQk"',
        "newrelic": "eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE5ODI2OTciLCJhcCI6IjE0MDk1Mjc5OCIsImlkIjoiZDlmNmMyNjA4MTAyNzk5NCIsInRyIjoiZjMwOGExYWI0YWE4ODMyM2M4YTNkMjk1MjhjNmM1NzAiLCJ0aSI6MTcxOTIyMzYxMTg0NSwidGsiOiIyNTMwMCJ9fQ==",
        "priority": "u=1, i",
        "referer": "https://www.bloomberg.com/search?query=NVDA",
        "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "traceparent": "00-f308a1ab4aa88323c8a3d29528c6c570-d9f6c26081027994-01",
        "tracestate": "25300@nr=0-1-1982697-140952798-d9f6c26081027994----1719223611845",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
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
        ticker_df.to_csv(f'ticker_data/news/{ticker}.csv')
        print('Export done.')


get_news()
