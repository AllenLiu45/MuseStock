from Model import *
from utils import *
import pickle
import torch
from torch import optim
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
from my_parser import args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(DEVICE):
    print(f'dataset:{args.dataset}')
    with open(f'./data/{args.dataset}/pkl/market.pkl', 'rb') as f:
        markets = pickle.load(f)
    with open(f'./data/{args.dataset}/pkl/labels.pkl', 'rb') as f:
        y_load = pickle.load(f)
    with open(f'./data/{args.dataset}/pkl/news.pkl', 'rb') as f:
        news = pickle.load(f)
    with open(f'./data/{args.dataset}/pkl/sentiment.pkl', 'rb') as f:
        sentiments = pickle.load(f)

    markets = markets.astype(np.float32)

    markets = torch.tensor(markets, device=DEVICE, dtype=torch.float32)

    news = torch.tensor(news, device=DEVICE, dtype=torch.float32)

    y = torch.tensor(y_load, device=DEVICE, dtype=torch.long)

    sentiments = torch.tensor(sentiments, device=DEVICE, dtype=torch.float32)

    if args.relation != "None":
        with open('./relations/' + args.relation + '_relation.pkl', 'rb') as handle:
            relation_static = pickle.load(handle)
        relation_static = torch.tensor(relation_static, device=DEVICE, dtype=torch.float32)
    else:
        relation_static = None


    return markets, news, y, sentiments, relation_static

def train(model, x_markets_train, x_news_train, x_sentiments_train, y_train, relation_static = None):
    model.train()
    seq_len = len(x_markets_train)
    train_seq = list(range(seq_len))[look_back_window:]
    total_loss = 0
    total_loss_count = 0
    batch_size = args.batch_size

    for i in train_seq:
        output = model(x_markets_train[i - look_back_window + 1: i + 1],
                       x_news_train[i - look_back_window + 1: i + 1],
                       x_sentiments_train[i],
                       relation_static = relation_static)

        loss = criterion(output, y_train[i])
        loss.backward()

        total_loss += loss.item()
        total_loss_count += 1

        if total_loss_count % batch_size == batch_size - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()


    if total_loss_count % batch_size != batch_size - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return total_loss / total_loss_count

def test(model, x_markets_eval, x_news_eval, x_sentiments_eval, y_eval, relation_static = None):
    model.eval()
    seq_len = len(x_markets_eval)
    seq = list(range(seq_len))[look_back_window:]

    preds = []
    trues = []

    for i in seq:
        output = model(x_markets_eval[i - look_back_window + 1: i + 1],
                       x_news_eval[i - look_back_window + 1: i + 1],
                       x_sentiments_eval[i],
                       relation_static = relation_static
                       )

        output = output.detach().cpu()
        preds.append(output.numpy())
        trues.append(y_eval[i].cpu().numpy())

    acc, mcc, f1 = metrics(trues, preds)
    return acc,  mcc, f1


if __name__=="__main__":

    set_seed(args.seed)
    DEVICE = "cuda:" + args.device
    criterion = torch.nn.CrossEntropyLoss()

    if args.relation != "None":
        static = 1
        pass
    else:
        static = 0
        relation_static = None

    # load dataset
    print("loading dataset")
    x_markets, x_news, y, sentiments, relation_static = load_dataset(DEVICE)

    # hyper-parameters
    num_stock = x_markets.size(1)
    D_MARKET = x_markets.size(2)
    D_NEWS = x_news.size(2)
    max_epoch =  args.max_epoch
    infer = args.infer
    hidn_rnn = args.hidn_rnn
    attention_heads = args.heads
    hidn_att= args.hidn_att
    lr = args.lr
    look_back_window = args.look_back_window
    dropout = args.dropout
    t_mix = 1

    print(f"num_stock:{num_stock}")
    print(f"d_market:{D_MARKET}")
    print(f"d_news:{D_NEWS}")
    print(f"attention_heads:{attention_heads}")
    print(f"look_back_window:{look_back_window}")
    print(f"hidn_rnn:{hidn_rnn}")
    print("======================")

    x_markets_train = x_markets[:-90]
    x_markets_test = x_markets[-90 - look_back_window:]

    y_train = y[:-90]
    y_test = y[-90 - look_back_window:]

    x_news_train = x_news[:-90]
    x_news_test = x_news[-90 - look_back_window:]

    x_sentiments_train = sentiments[:-90]
    x_sentiments_test = sentiments[-90 - look_back_window:]

    # initialize
    best_model_file = 0
    epoch = 0
    patience = 0
    test_acc_best = 0
    test_mcc_best = 0
    testa_f1_best = 0


    model = MuseStock(num_stock=num_stock, d_market = D_MARKET,d_news= D_NEWS,
                    d_hidden = D_MARKET, hidn_rnn = hidn_rnn, heads = attention_heads,
                    hidn_att= hidn_att, dropout = dropout,t_mix = t_mix, relation_static = static)

    model.cuda(device=DEVICE)
    model.to(torch.float)

    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_constraint)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)  # 余弦退火学习率

    #train
    for epoch in tqdm(range(max_epoch)):
        train_loss = train(model, x_markets_train, x_news_train, x_sentiments_train, y_train, relation_static = relation_static)
        scheduler.step()
        test_acc, test_mcc, test_f1 = test(model, x_markets_test, x_news_test, x_sentiments_test, y_test, relation_static = relation_static)
        print(f"epoch {epoch + 1}, train_loss:{train_loss:.4f}, test_acc:{test_acc:.4f}, test_mcc:{test_mcc:.4f}, test_f1:{test_f1:.4f}")

        if test_acc > test_acc_best:
            test_acc_best = test_acc
            test_mcc_best = test_mcc
            test_f1_best = test_f1
            patience = 0

            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = f"./result/model_saved/acc{test_acc_best:.4f}_mcc{test_mcc_best:.4f}_f1{test_f1_best:.4f}"
                torch.save(model.state_dict(), best_model_file)
        else:
            patience += 1

        if patience > args.patience:
            with open(f'./result/test_result/lr:{args.lr}_rnn:{args.look_back_window}_heads:{args.heads}_hidn:{args.hidn_att}_dropout:{args.dropout}.txt','w') as file:
                file.write(f"Evaluation on epoch {epoch}:\n")
                file.write(f"Train Loss: {train_loss}\n")
                file.write(f"lr: {args.lr}\n")
                file.write(f"look_back_window: {args.look_back_window}\n")
                file.write(f"heads_att: {args.heads}\n")
                file.write(f"hidn_att: {args.hidn_att}\n")
                file.write(f"dropout: {args.dropout}\n\n\n")
                file.write(f"Test Accuracy: {test_acc_best}\n")
                file.write(f"Test MCC: {test_mcc_best}\n")
            print(f"saved_model_result: test_acc:{test_acc_best:.4f}, test_mcc:{test_mcc_best:.4f}, test_f1:{test_f1_best:.4f}")
            break
        epoch += 1
