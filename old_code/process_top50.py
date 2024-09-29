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
import ast
import json

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import os
import shutil
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

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
# with_sentiment_dir = 'with_sentiment_content/'
col_name = 'Article_title'
with_sentiment_dir = 'with_sentiment/'
top50_dir = directory + 'top50_news/'
top50 = os.listdir(top50_dir)
# top50 = [item for item in top50 if
#          item not in os.listdir(top50_dir + with_sentiment_dir) and
#          item.endswith('.csv')]

# news_w_content_dir = directory + 'top50_news_w_content/'
# top50_w_content = os.listdir(news_w_content_dir)
# top50_w_content.remove('with_sentiment')
# top50_w_content = [item for item in top50_w_content if item not in os.listdir(news_w_content_dir + with_sentiment_dir)]
#
# news_headlines_only = directory + 'top50_news_headlines_only/'
# top50_headlines = os.listdir(news_headlines_only)
# top50_headlines.remove('with_sentiment')
# top50_headlines = [item for item in top50_headlines if item not in os.listdir(news_headlines_only + with_sentiment_dir)]

#
# model_name = "ProsusAI/finbert"
# sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, top_k=None,
#                               max_length=512, truncation=True, device='cuda:0')
#
# for i, stock in enumerate(top50):
#     print(f'Processing {stock} ({i})...')
#
#     features_list = ['Date', 'Article_title', 'Stock_symbol', 'Url']
#     df = pd.read_csv(top50_dir + stock)
#     df.dropna(subset=['Article_title'], inplace=True)
#     df = df[features_list]
#
#     try:
#         # dataset = load_dataset('csv', data_files=top50_dir + stock, column_names=features_list)
#         dataset = Dataset.from_pandas(df)
#         print(f'{stock} has ' + str(len(dataset)) + ' samples')
#     except ValueError as e:
#         print(f'{stock} has no data. Skipping...')
#         continue
#
#     # dataset = dataset.map(get_sentiment)
#     # dataset.to_csv(top50_dir + with_sentiment_dir + stock, index=False)
#
#     out = sentiment_pipeline(KeyDataset(dataset, col_name), batch_size=256)
#     df_out = pd.DataFrame(out)
#     assert len(df_out) == len(df)
#
#     to_concat = [df]
#     for col in df_out.columns:
#         label_name = pd.json_normalize(df_out[col]).iloc[0]['label']
#         to_concat.append(pd.json_normalize(df_out[col])[['score']].rename(columns={'score':label_name}))
#
#     result_df = pd.concat(to_concat, axis=1)
#     result_df.to_csv(directory + 'top50_news/' + with_sentiment_dir + stock, index=False)


# emotion classification
top50_dir = directory + 'top50_news_w_content/with_sentiment/'
top50 = os.listdir(top50_dir)
output_directory = directory + 'top50_news_w_content/with_emotion_content/'

model_name = "j-hartmann/emotion-english-distilroberta-base"
sentiment_pipeline = pipeline("text-classification", model=model_name, top_k=None, device='cuda:0')

for i, stock in enumerate(top50):
    print(f'Processing {stock} ({i})...')

    features_list = ['Date', 'Article_title', 'Stock_symbol', 'Url', 'Lsa_summary', 'neutral', 'positive', 'negative']
    df = pd.read_csv(top50_dir + stock)
    df = df[features_list]

    try:
        # dataset = load_dataset('csv', data_files=top50_dir + stock, column_names=features_list)
        dataset = Dataset.from_pandas(df)
        print(f'{stock} has ' + str(len(dataset)) + ' samples')
    except ValueError as e:
        print(f'{stock} has no data. Skipping...')
        continue

    # dataset = dataset.map(get_sentiment)
    # dataset.to_csv(top50_dir + with_sentiment_dir + stock, index=False)

    out = sentiment_pipeline(KeyDataset(dataset, col_name), batch_size=256)
    df_out = pd.DataFrame(out)
    assert len(df_out) == len(df)

    to_concat = [df]
    for col in df_out.columns:
        label_name = pd.json_normalize(df_out[col]).iloc[0]['label']
        if label_name == 'neutral':
            label_name = 'neutral_emotion'
        to_concat.append(pd.json_normalize(df_out[col])[['score']].rename(columns={'score':label_name}))

    result_df = pd.concat(to_concat, axis=1)
    result_df.to_csv(output_directory + stock, index=False)