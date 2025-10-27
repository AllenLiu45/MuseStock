import os
import pandas as pd
import shutil
from datetime import datetime
import pickle
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from my_parser import args

def generate_ticker_list(price_path, news_path, save_path):
    ticker_csv_list = []
    for ticker_csv in os.listdir(price_path):
        if ticker_csv.endswith(".csv"):
            ticker = ticker_csv.split(".")[0]
            ticker_csv_list.append(ticker)

    news_list = []
    for news in os.listdir(news_path):
        news_list.append(news)

    for ticker_csv in ticker_csv_list:
        if ticker_csv not in news_list:
            print(ticker_csv)

    for news in news_list:
        if news not in ticker_csv_list:
            print(news)


    df = pd.DataFrame(ticker_csv_list, columns=['ticker'])
    df.to_csv(f"{save_path}ticker_csv_list.csv", index=False)

def generate_trading_date_list(price_path, save_path):
    df = pd.read_csv(price_path)

    ticker_csv_list = df['Date'].tolist()

    df = pd.DataFrame(ticker_csv_list, columns=['Date'])
    df.to_csv(f"{save_path}trading_date_list.csv", index=False)
    print(len(ticker_csv_list))


def read_pkl(dataset):
    with open(f"./data/{dataset}/pkl/market.pkl", "rb") as f:
        market_data = pickle.load(f)
        market_data = market_data.astype(np.float64)
        market_data = torch.tensor(market_data)
        print(market_data.shape)

    with open(f"./data/{dataset}/pkl/news.pkl", "rb") as f:
        news_data = pickle.load(f)
        news_data = news_data.astype(np.float64)
        news_data = torch.tensor(news_data)
        print(news_data.shape)

    with open(f"./data/{dataset}/pkl/labels.pkl", "rb") as f:
        labels = pickle.load(f)
        labels = labels.astype(np.float64)
        labels = torch.tensor(labels)
        print(labels.shape)

    with open(f"./data/{dataset}/pkl/sentiment.pkl", "rb") as f:
        sentiment = pickle.load(f)
        sentiment = sentiment.astype(np.float64)
        sentiment = torch.tensor(sentiment)
        print(sentiment[0][2])

def check_date(price_path, news_path):


    ticker_date_map = {}
    for ticker_csv in os.listdir(price_path):
        if ticker_csv.endswith(".csv"):
            source_path = os.path.join(price_path, ticker_csv)
            df = pd.read_csv(source_path)

            ticker = ticker_csv.replace(".csv", "")

            start_date = df['Date'].iloc[0]
            end_date = df['Date'].iloc[-1]

            ticker_date_map[ticker] = (start_date, end_date)

    date_ticker_map = defaultdict(list)

    for ticker, (start_date, end_date) in ticker_date_map.items():
        date_ticker_map[(start_date, end_date)].append(ticker)


    for (start_date, end_date), tickers in sorted(date_ticker_map.items(), key=lambda x: x[0]):
        print(f"price Tickers: {tickers}, Start Date: {start_date}, End Date: {end_date}")

    ticker_news_date_map = {}


    for ticker_folder in os.listdir(news_path):
        ticker_folder_path = os.path.join(news_path, ticker_folder)
        if os.path.isdir(ticker_folder_path):

            csv_files = [f for f in os.listdir(ticker_folder_path) if f.endswith(".csv")]
            if csv_files:

                dates = [datetime.strptime(f.split('.')[0], "%Y-%m-%d") for f in csv_files]

                start_date = min(dates).strftime("%Y-%m-%d")
                end_date = max(dates).strftime("%Y-%m-%d")
            else:
                start_date = "9999-12-31"
                end_date = "9999-12-31"


            ticker_news_date_map[ticker_folder] = (start_date, end_date)

    print('\n')

    news_date_ticker_map = defaultdict(list)

    for ticker, (start_date, end_date) in ticker_news_date_map.items():
        news_date_ticker_map[(start_date, end_date)].append(ticker)

    for (start_date, end_date), tickers in sorted(news_date_ticker_map.items(), key=lambda x: x[0]):
        print(f"News Tickers: {tickers}, Start Date: {start_date}, End Date: {end_date}")

    print('\n')

    for ticker in ticker_date_map.keys():
        if ticker not in ticker_news_date_map.keys():
            print(f"{ticker} is not in news folder")
    print('\n')
    for ticker in ticker_news_date_map.keys():
        if ticker not in ticker_date_map.keys():
            print(f"{ticker} is not in price folder")

def test_sentiment():
    local_model_path = "your baseline_model path"
    device = torch.device("cuda:1")

    bert_model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    sentence = "医药生物行业:注射剂一致性评价来临,利好高质量龙头公司。医疗保健：同时明确首家生物类似药为高 荐14股。医药行业2018年1月份投资月报:2018的“2+2”医药投资策略。【医药】江琦：医药生物行业2018年1月月报：2018是政策落地大年，看好创新药、高品质仿制药、医疗服务三大方向-增持-0101。"
    vectors = tokenizer(sentence, padding=True, max_length=2048, return_tensors='pt').to(device)
    outputs = bert_model(**vectors).logits
    probs = torch.nn.functional.softmax(outputs, dim=1)
    print(probs)
    bert_dict = {
        'neg': round(probs[0][0].item(), 3),
        'neu': round(probs[0][1].item(), 3),
        'pos': round(probs[0][2].item(), 3)
    }

    max_prob = max(bert_dict.values())
    max_category = [category for category, p in bert_dict.items() if p == max_prob][0]
    print(max_category)

def read_csv(path):
    df = pd.read_csv(path)
    print(df.tail(10))

if __name__ == "__main__":

    check_date(price_path = f"./data/{args.dataset}/price/",
               news_path = "./data/{args.dataset}/news_original/")

    generate_ticker_list(price_path=f"../data/{args.dataset}/price/",
                         news_path=f"../data/{args.dataset}/news_original/",
                         save_path=f"../data/{args.dataset}/")

    generate_trading_date_list(price_path=f"../data/{args.dataset}/price/东方财富.csv",
                               save_path=f"../data/{args.dataset}/")

    # read_pkl(dataset="CMIN-CN")

    # read_npy()

    # test_sentiment()

