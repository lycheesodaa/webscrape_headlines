import os
import time

import pandas as pd
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from collections import defaultdict
from sumy.utils import get_stop_words

# Initialize LexRankSummarizer and Tokenizer
stemmer = Stemmer("english")
summarizer = LsaSummarizer(stemmer)
tokenizer = Tokenizer("english")
summarizer.stop_words = get_stop_words("english")

directory = 'external_data/FNSPID/'
news_directory = directory + 'full_news/'
hist_directory = directory + 'full_history/'

top50_dir = directory + 'top50_news/'
top50 = os.listdir(top50_dir)


def increase_weight_for_key_words(sentences, key_words):
    sentence_weights = defaultdict(float)

    for sentence in sentences:
        for word in key_words:
            if word.lower() in str(sentence).lower():
                sentence_weights[sentence] += 1
    return sentence_weights


def new_sum(text, key_words, num_sentences):
    parser = PlaintextParser.from_string(text, tokenizer)
    initial_summary = summarizer(parser.document, num_sentences)
    # Increase weight
    sentence_weights = increase_weight_for_key_words(parser.document.sentences, key_words)

    # Combine weights from initial summary with additional weights
    for sentence in initial_summary:
        sentence_weights[sentence] += 1  # Initial summary sentences get additional weight

    # Select top sentences as final summary
    final_summary = sorted(sentence_weights, key=sentence_weights.get, reverse=True)[:num_sentences]

    # Output final summary
    final_summary_text = " ".join(str(sentence) for sentence in final_summary)

    return final_summary_text


def from_csv_summarize(folder_path, saving_path):
    a = time.time()
    # 遍历每个CSV文件
    for csv_file in top50:
        file_path = os.path.join(top50_dir, csv_file)

        # 读取CSV文件
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="Windows-1252")
        symbol = csv_file.split(".")[0].upper()

        # Check if null summaries exist - if true, extract article and summarise

        # df.columns = df.columns.str.capitalize()
        key_words_value = {symbol}
        num_sentences_value = 3
        # text_ratio = 0.25

        # 对text列进行操作，这里简单示例为将text列的内容转换为大写
        df['Lsa_summary_2'] = df['Article'].apply(new_sum, key_words=key_words_value, num_sentences=num_sentences_value)
        # df['New_text'] = df['Text'].apply(summarize, num_sentences=num_sentences_value)

        # 删除多余行，避免冗余
        df = df.drop(columns=['Text'])

        # 删除mark列不为1的行
        df = df[df['Mark'] == 1]
        print(time.time()-a, "s")

        # 保存修改后的DataFrame到CSV文件
        df.to_csv(os.path.join(saving_path, csv_file), index=False)


from_csv_summarize(top50_dir, top50_dir)
