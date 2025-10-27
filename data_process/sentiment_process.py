import os
import numpy as np
from modelscope import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import torch
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
from warnings import filterwarnings
filterwarnings("ignore")
from my_parser import args

def sentiment_for_relation(csv_news_path, news_sentiment_path, trading_date_list, local_model_path):
    device = torch.device("cuda:1")

    model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)


    df_date = pd.read_csv(trading_date_list)

    for date in tqdm(df_date['Date']):
        for ticker in os.listdir(csv_news_path):
            old_ticker_path = os.path.join(csv_news_path, ticker)
            new_ticker_path = os.path.join(news_sentiment_path, ticker)
            os.makedirs(new_ticker_path, exist_ok=True)

            sentiment_path = os.path.join(new_ticker_path, f"{date}.npy")
            csv_news_date_path = os.path.join(old_ticker_path, f"{date}.csv")

            if os.path.exists(csv_news_date_path):
                df = pd.read_csv(csv_news_date_path)
                texts = df['text'].dropna().astype(str).tolist()

                total_probs = np.zeros(3)
                count = 0

                for sentence in texts:
                    inputs = tokenizer(sentence, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
                    outputs = model(**inputs).logits
                    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[0]

                    total_probs += probs
                    count += 1


                if count > 0:
                    avg_probs = total_probs / count
                else:
                    avg_probs = np.array([0, 1, 0])

                np.save(sentiment_path, avg_probs)
            else:
                np.save(sentiment_path, np.array([0, 1, 0]))


if __name__ == '__main__':
    # dataset_name = "CMIN-US" or "ACL18
    if args.dataset_name == "CMIN-US" or "ACL18":
        sentiment_for_relation(csv_news_path="../data/CMIN-US/news_original/",
                               news_sentiment_path="../data/CMIN-US/sentiment_for_relation/",
                               trading_date_list="../data/CMIN-US/trading_date_list.csv",
                               local_model_path='your baseline_model path',
                               )

    # dataset_name = "CMIN-CN"
    if args.dataset_name == "CMIN-CN":
        sentiment_for_relation(csv_news_path="../data/CMIN-CN/news_original/",
                               news_sentiment_path="../data/CMIN-CN/sentiment_for_relation/",
                               trading_date_list="../data/CMIN-CN/trading_date_list.csv",
                               local_model_path='your  baseline_model path')





