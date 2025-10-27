import time
from transformers import BertTokenizer, BertModel
from modelscope import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import warnings
import os
import pandas as pd
import json
import csv
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
from my_parser import args
warnings.filterwarnings("ignore")


news_dim = 20

def ACL_News_Process(raw_news_path, csv_news_path):
    def file_to_csv(raw_news_path, csv_news_path):
        for ticker in os.listdir(raw_news_path):
            old_ticker_path = os.path.join(raw_news_path, ticker)
            new_ticker_path = os.path.join(csv_news_path, ticker)
            if not os.path.exists(new_ticker_path):
                os.makedirs(new_ticker_path)
            for date in os.listdir(old_ticker_path):
                old_news_date_path = os.path.join(old_ticker_path, date)
                new_news_date_path = os.path.join(new_ticker_path, f"{date}.csv")
                with open(old_news_date_path, 'r',encoding='utf-8') as infile, open(new_news_date_path, 'w', newline='',encoding='utf-8') as outfile:

                    csv_writer = csv.writer(outfile)

                    csv_writer.writerow(['text'])


                    for line in infile:
                        try:

                            data = json.loads(line)

                            text = data.get('text', '')

                            csv_writer.writerow([text])
                        except json.JSONDecodeError:
                            print(f"Failed: {ticker} {date}")


    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'URL', '', text)
        text = re.sub(r'AT_USER', '', text)
        text = re.sub(r'- - ', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_data(csv_news_path):
        for parent_file in os.listdir(csv_news_path):
            for filename in os.listdir(os.path.join(csv_news_path, parent_file)):
                if filename.endswith(".csv"):
                    try:
                        daily_path = os.path.join(csv_news_path, parent_file, filename)
                        daily_df = pd.read_csv(daily_path)
                        daily_df['text'] = daily_df['text'].apply(clean_text)
                        daily_df.to_csv(daily_path, index=False)
                    except:
                        print(f"{parent_file},{filename}failed")
                        continue

                print(f"{parent_file},{filename}finished")
            print(f"{filename}finished")

    file_to_csv(raw_news_path, csv_news_path)
    clean_data(csv_news_path)

def CMIN_News_Process(old_path, new_path, language='CN'):
    for ticker in os.listdir(old_path):
        for date in os.listdir(os.path.join(old_path, ticker)):
            input_file = os.path.join(old_path, ticker, date)
            output_file = os.path.join(new_path, ticker, f'{date}.csv')

            with open(input_file, "rb") as f:

                lines = f.readlines()


            processed_sentences = []

            for line in lines:
                try:

                    data = json.loads(line)
                    if "text" in data:

                        if language == 'CN':
                            sentence = "".join(data["text"]).replace("\u2019", "'").replace("\u00e9", "é") + "。"
                        else:
                            sentence = "".join(data["text"]).replace("\u2019", "'").replace("\u00e9", "é")
                        processed_sentences.append(sentence)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
                    break


            if not os.path.exists(output_file):

                output_dir = os.path.dirname(output_file)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(output_file, 'w') as f:
                    pass


            df = pd.DataFrame(processed_sentences, columns=["text"])
            df.to_csv(output_file, index=False)

            print(f"{ticker}:{date}.csv saved successfully")

class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, bidirectional=True):
        super(BiGRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Bi-GRU
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        output, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim * 2)
        return output

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Bi-LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, x):

        output, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # Q, K, V
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_k = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_v = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        Q = self.W_q(x)  # (batch_size, seq_len, hidden_dim * 2)
        K = self.W_k(x)  # (batch_size, seq_len, hidden_dim * 2)
        V = self.W_v(x)  # (batch_size, seq_len, hidden_dim * 2)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(
            self.hidden_dim * 2)  # (batch_size, seq_len, seq_len)


        attn_weights = self.softmax(attn_scores)  # (batch_size, seq_len, seq_len)


        context_vector = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim * 2)


        output = torch.sum(context_vector, dim=1)  # (batch_size, hidden_dim * 2)

        return output


class NewsEmbeddingModel(nn.Module):
    def __init__(self, encoder_type="gru", use_attention=True, input_dim=768, hidden_dim=256, output_dim=news_dim):
        super(NewsEmbeddingModel, self).__init__()

        # 选择 Bi-GRU 或 Bi-LSTM
        if encoder_type == "gru":
            self.encoder = BiGRUEncoder(input_dim, hidden_dim)
        elif encoder_type == "lstm":
            self.encoder = BiLSTMEncoder(input_dim, hidden_dim)
        else:
            raise ValueError("Invalid encoder_type. Choose 'gru' or 'lstm'.")

        self.use_attention = use_attention
        if use_attention:
            self.attention = ScaledDotProductAttention(hidden_dim)
        else:
            self.attention = BiLSTMEncoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)

        # 两次 MLP 降维
        self.dim_reducer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )



    def forward(self, x):
        encoded_output = self.encoder(x)  # (batch_size, seq_len, hidden_dim * 2)

        if self.use_attention:
            context_vector = self.attention(encoded_output)  # (batch_size, hidden_dim * 2)
        else:
            context_vector = self.attention(encoded_output)[:, -1, :]

        return self.dim_reducer(context_vector)  # (batch_size, output_dim)

def generate_zero_vector_and_save(ticker, date, embedding_path):
    zero_vector = np.zeros(news_dim)
    embedding_file_path = os.path.join(embedding_path, ticker, f"{date}.npy")
    os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
    np.save(embedding_file_path, zero_vector)

def get_news_embedding(csv_news_path, embedding_path, local_model_path, trading_date_list,encoder_type="gru", use_attention=True, look_back_days=5):
    print(datetime.now())
    device = torch.device("cuda:0")


    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    bert_model = AutoModel.from_pretrained(local_model_path).to(device)
    bert_model.eval()


    news_model = NewsEmbeddingModel(encoder_type=encoder_type, use_attention=use_attention).to(device)
    news_model.eval()


    df_date = pd.read_csv(trading_date_list)

    for date in tqdm(df_date['Date']):
        for ticker in os.listdir(csv_news_path):
            old_ticker_path = os.path.join(csv_news_path, ticker)
            new_ticker_path = os.path.join(embedding_path, ticker)
            os.makedirs(new_ticker_path, exist_ok=True)

            embedding_file_path = os.path.join(new_ticker_path, f"{date}.npy")

            past_news_embeddings = []
            for i in range(look_back_days + 1):
                past_date_index = df_date[df_date['Date'] == date].index[0] - i
                if past_date_index < 0:
                    continue
                past_date = df_date.iloc[past_date_index]['Date']
                csv_news_date_path = os.path.join(old_ticker_path, f"{past_date}.csv")

                if os.path.exists(csv_news_date_path):
                    df = pd.read_csv(csv_news_date_path)
                    sentence = '. '.join(df['text'].dropna().astype(str))
                    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                    sentence_embedding = outputs.last_hidden_state[:, 0, :]
                    past_news_embeddings.append(sentence_embedding.cpu().numpy())

            if len(past_news_embeddings) == 0:
                generate_zero_vector_and_save(ticker, date, embedding_path)
                continue

            past_news_embeddings = past_news_embeddings[::-1]

            past_news_embeddings = np.stack(past_news_embeddings, axis=1)
            past_news_embeddings = torch.tensor(past_news_embeddings, dtype=torch.float32).to(device)
            with torch.no_grad():
                final_embedding = news_model(past_news_embeddings)

            np.save(embedding_file_path, final_embedding.cpu().numpy())
def check_embedding():
    path = f"../data/{args.dataset}/news_embedding/XOM/2018-01-03.npy"
    embedding = np.load(path)
    print(embedding)

if __name__ == "__main__":
    if args.dataset == "ACL18":
        ACL_News_Process(raw_news_path=f"../data/ACL18/news/preprocessed/",
                         csv_news_path="../data/ACL18/news_original/",)

        # get news embedding
        get_news_embedding(csv_news_path="../data/ACL18/news_original/csv/",
                           embedding_path="../data/ACL18/news_embedding/",
                           local_model_path="your baseline_model path",
                           trading_date_list="../data/ACL18/trading_date_list.csv",
                           encoder_type="gru",
                           use_attention=True)


    if args.dataset == "CMIN-US":
        CMIN_News_Process(raw_news_path="../data/CMIN-US/news/preprocessed/",
                         csv_news_path="../data/CMIN-US/news_original/",
                          language="US")

        # get news embedding
        get_news_embedding(csv_news_path="../data/CMIN-US/news_original/",
                           embedding_path="../data/CMIN-US/news_embedding/",
                           local_model_path="your baseline_model path",
                           trading_date_list="../data/CMIN-US/trading_date_list.csv",
                           encoder_type="gru",
                           use_attention=True,
                           )

    if args.dataset == "CMIN-CN":
        CMIN_News_Process(raw_news_path="../data/CMIN-CN/news/preprocessed/",
                         csv_news_path="../data/CMIN-CN/news_original/",
                          language="CN")

        get_news_embedding(csv_news_path="../data/CMIN-CN/news_original/",
                       embedding_path="../data/CMIN-CN/news_embedding/",
                       local_model_path="your baseline_model path",
                       trading_date_list="../data/CMIN-CN/trading_date_list.csv",
                       encoder_type="gru",
                       use_attention=True,
                       )


