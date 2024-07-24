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
    file_list = os.listdir("yfinance_ticker_data/news")
    file_names_without_extension = [os.path.splitext(f)[0] for f in file_list]

    return file_names_without_extension


def iterate_pages(ticker, security):
    page = 1

    url = f"https://seekingalpha.com/api/v3/symbols/{ticker}/news?filter[since]=0&filter[until]=0&id=nvda&include=author%2CprimaryTickers%2CsecondaryTickers%2Csentiments&isMounting=true&page[size]=50&page[number]="

    payload = {}
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cookie': 'machine_cookie=2239684902777; _gcl_au=1.1.924621184.1719286057; _ga=GA1.1.490201936.1719286057; _pcid=%7B%22browserId%22%3A%22lxtuj0qz153wfql9%22%7D; __pat=-14400000; pxcts=e1fc746f-32a2-11ef-89b5-cd5723fe6abc; _pxvid=e1fc66cf-32a2-11ef-89b5-711f58330ef6; hubspotutk=cad02acd2ec4b91c297eb6e96618a316; __hssrc=1; _hjSessionUser_65666=eyJpZCI6IjU4NTRlMzQ2LTQyZTItNWI3Mi05NDk2LTVkODQxMDg1YzEzNyIsImNyZWF0ZWQiOjE3MTkyODYwNTc0NDUsImV4aXN0aW5nIjp0cnVlfQ==; _hjSession_65666=eyJpZCI6Ijc2OTYyYTE3LThmNjItNDczZi1iMzcwLTI4ZGJhNTdkZjdhYSIsImMiOjE3MjExODI2ODMxNzAsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; g_state={"i_p":1721189909038,"i_l":1}; _igt=9aec0d4c-ebe1-4da6-9507-3931edaeb886; _sapi_session_id=F9ERvQqyXoO806yJp%2FLZgXDM4tAwCGvPqbXb9mX%2FvlIDlDu0oUyNAjpc57vNArYe%2FmruMc0gbZ8%2BEe%2B5mbBQTxnVw4%2BsNer8uneO7f8Z2jWRSJSNTrs8H1TAxjFIImFH%2FpBgrUMy7jfM24e%2BytIWTPx8kuAyU77WNF4QcUvnF%2FcPQbpztQ2RHIhG6aXVlkf5ISviw8JHJqzQifQmQAGgcbLhoA29VF1zLIQP9WrxNoC1xBiyAmVtUB2SWRSOaBu7hjpqaMDVhPimlMGukflFbumQ2XAmS6GL6IpQykZFCahuN6GNyMUhPt5abyHlECYYOHZxAZdiHMhl1j7J6pvjaRXuf18rii6d65rLaexWk85XEbN4P1A%2BZjEYaj1FfGph1F4%2BR%2FvImlU%3D--A06%2FtBP%2Fuqp0snj7--%2BIjMxnBuPjg%2By3I0z2DE7g%3D%3D; sailthru_hid=3a6c0dbb25c63193402d27c1d0d1c20066972a6d958940239603092f14f71a3d3d86b110bdcdbc15ec283aee; session_id=815c84e7-9297-471f-9875-e40e42cc91ce; _sasource=; _ig=60674061; __hstc=234155329.cad02acd2ec4b91c297eb6e96618a316.1719286059889.1721182686574.1721185884030.3; _pctx=%7Bu%7DN4IgrgzgpgThIC5gF8A05owMoBcCGOkiIeAdgPakjoQCWOUAkgCaKlgA2HyQA; LAST_VISITED_PAGE=%7B%22pageKey%22%3A%2235d0629a-b4ac-4c2e-b458-4271d851a8b0%22%2C%22pathname%22%3A%22https%3A%2F%2Fseekingalpha.com%2Fsymbol%2FNVDA%2Fnews%22%7D; sailthru_pageviews=4; _hjHasCachedUserAttributes=true; __pvi=eyJpZCI6InYtMjAyNC0wNy0xNy0xMC0xOC0wMi05MjAtMEw4Q0ZKVFpYMkhoQ1R5Yi1hYzZhNDE1NjdmMDI0YTA4ZWU5YzZiMTdjODgzYTdhZiIsImRvbWFpbiI6Ii5zZWVraW5nYWxwaGEuY29tIiwidGltZSI6MTcyMTE4ODAyMDkwMX0%3D; sailthru_visitor=edcb5b43-395d-4590-ac0f-c85f24000dca; __tbc=%7Bkpex%7DIHNKVOty_QaVK_sPnVz_3sGywXeHangogroIhDGCRyEwA75AfFmA_eSFgY7p3f_X; xbc=%7Bkpex%7DP6h1xjsxZeFrDeK3lcJWcPnSp69gvqSXT6vBBf3xZqr7W5KNXQiQMi3AGsp3Vv05DaWts_WC1Fxa0UXEbx58ElJWp7-fAtPGdpSp7ZppNIDv6x0a54RioaXzG9XeMsToeF6e0S2FDeOlrDHqqq9VzOLWq_5STf1C7FC9kOrwQ_A; __hssc=234155329.4.1721185884030; _px3=b4c1118e0644f4aae956f9a57bbedd979b3224c04dae484e00cff6a67eaf9c58:yZ9Lj4dwmyhq4EB80f9crJXm5L/E8s+xAutynohoa3zKUePk5ffEQl11Lo57PHAnk+OVTJYPnC27I6DxrE2svA==:1000:Zr6+6h9GYQatq9P+jNem87i1Avx9roePdOVIqQt/hBbXLChmoosIkE1jwwsSvMfqKgr3H5Ioi0ZihnGM+fPRzEGynpMuuzLVp5z2UO90VH+pgVglWmz0tFqK16akR7ycVQejOOF3AUiGQU8DZ5b2x66lNZsXrN7FdktcbzPkfeVhcVUzLQNuMpoKvLCsb26OdSpL6k5eRcIMzIwPwteIZ9YKwrqXnyBa5Yv1Rt7XsIw=; sailthru_content=966116dee0094570e19e54cf059318bedcfd6eb540394c0d66a1e5cad7b34e066d033c8f7ec9accb700e427f77fffe9e3c5f57e561d9315a8a9a86be8720511eb52624f3aa6be2d685985686dd2b4580b16d356c09d8b6c269b0992bfc493091; _pxde=b836b99d1aa61d9b706310cc9f40b343afe2af3eb07a88182db75e865139fedc:eyJ0aW1lc3RhbXAiOjE3MjExODgwNDA3NTIsImZfa2IiOjB9; _ga_KGRFF2R2C5=GS1.1.1721182682.2.1.1721188221.32.0.0; userLocalData_mone_session_lastSession=%7B%22machineCookie%22%3A2239684902777%2C%22machineCookieSessionId%22%3A%222239684902777%261721182682337%22%2C%22sessionStart%22%3A1721182682337%2C%22sessionEnd%22%3A1721190023646%2C%22firstSessionPageKey%22%3A%22006fffd3-1b2e-41e5-bf09-9f8aa72eb26e%22%2C%22isSessionStart%22%3Afalse%2C%22lastEvent%22%3A%7B%22event_type%22%3A%22mousemove%22%2C%22timestamp%22%3A1721188223646%7D%7D',
        'priority': 'u=1, i',
        'referer': 'https://seekingalpha.com/symbol/NVDA/news?page=10',
        'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
    }

    response = requests.request("GET", url + str(page), headers=headers, data=payload)

    # fencepost for the initial dataframe
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame.from_dict(data['results'])
    time.sleep(rd.randint(1, 5))
    page += 1
    print('Total news entries: {}'.format(data['total']))

    # iterate until 'results' segment is empty
    while True:
        response = requests.get(url + str(page), headers=headers)
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
