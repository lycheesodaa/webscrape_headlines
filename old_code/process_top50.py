# import pandas as pd
# from transformers import pipeline
# from huggingface_hub import login
# import bs4 as bs
# import requests
# from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoConfig
# import numpy as np
# from scipy.special import softmax
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import os
import shutil
from transformers import pipeline

load_dotenv()
# login(os.getenv('HF_TOKEN'), add_to_git_credential=True)

directory = 'external_data/FNSPID/'
news_directory = directory + 'full_news/'
hist_directory = directory + 'full_history/'


# def get_top_snp_symbols():
#     resp = requests.get('https://stockanalysis.com/list/sp-500-stocks/')
#     soup = bs.BeautifulSoup(resp.text, 'lxml')
#     table = soup.find_all('table')[0]  # Grab the first table
#
#     symbols = {}
#     for i, row in enumerate(table.findAll('tr')[1:]):
#         ticker = row.findAll('td')[1].text.strip('\n')
#         security = row.findAll('td')[2].text.strip('\n')
#         symbols[ticker] = security
#         if i == 200:
#             break
#     print(f"DEBUG: Total number of stock symbols: {len(symbols)}")
#
#     return symbols
#
#
# file_list = os.listdir(news_directory)
# top_few = get_top_snp_symbols()
# top_few = [stock + '.csv' for stock in top_few]
#
# intersection = [top for top in top_few if top in file_list]
# print(intersection)
# top50 = []
#
# # Get the top 50 stocks that have summaries
# for i, csv_file in enumerate(intersection):
#     try:
#         df = pd.read_csv(news_directory + csv_file)
#         column_list = ['Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']
#         if df[column_list].isnull().all().all():  # ignore stocks that do not have summaries
#             continue
#         top50.append(csv_file)
#         df.to_csv(directory + f'top50_news/{csv_file}', index=False)
#         if len(top50) == 50:
#             break
#     except FileNotFoundError as file_not_found:
#         print(file_not_found)
#
# print(top50)

# col_name = 'Lsa_summary'
col_name = 'Article_title'
with_sentiment_dir = 'with_sentiment/'
top50_dir = directory + 'top50_news/'
top50 = os.listdir(top50_dir)

news_w_content_dir = directory + 'top50_news_w_content/'
top50_w_content = os.listdir(news_w_content_dir)
top50_w_content.remove('with_sentiment')
top50_w_content = [item for item in top50_w_content if item not in os.listdir(news_w_content_dir + with_sentiment_dir)]

news_headlines_only = directory + 'top50_news_headlines_only/'
top50_headlines = os.listdir(news_headlines_only)
top50_headlines.remove('with_sentiment')
top50_headlines = [item for item in top50_headlines if item not in os.listdir(news_headlines_only + with_sentiment_dir)]


model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, top_k=None,
                              max_length=512, truncation=True)

# Function to perform sentiment analysis and append sentiment for scores to existing row
def get_sentiment(example):
    results = sentiment_pipeline(example[col_name])
    for sentiment in results[0]:
        example[sentiment['label']] = sentiment['score']
    return example

for i, stock in enumerate(top50_headlines):
    print(f'Processing {stock} ({i})...')
    try:
        dataset = load_dataset('csv', data_files=news_headlines_only + stock)
        print(f'{stock} has ' + str(len(dataset['train'])) + ' samples')
    except ValueError as e:
        print(f'{stock} has no data. Skipping...')
        continue

    dataset = dataset.map(get_sentiment)

    dataset['train'].to_csv(news_headlines_only + with_sentiment_dir + stock, index=False)

    # file copying utils
    # try:
    #     shutil.copyfile(hist_directory + stock, directory + f'top50_processed/{stock}')
    # except FileNotFoundError as e:
    #     print(e)
    #     shutil.copyfile(hist_directory + stock.lower(), directory + f'top50_processed/{stock}')

    # file modification utils
    # df = df[df['Article'].isna()]
    # print(f'Number of news headlines in {stock}: {len(df)}')
    # df.to_csv(news_headlines_only + stock, index=False)

