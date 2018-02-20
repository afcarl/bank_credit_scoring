import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict, one_hot_conversion
import pandas as pd
from collections import OrderedDict
from bidict import bidict
import torch
import random
import datetime as dt
from numpy import log



price_normalize = lambda x: (x.div(x.iloc[0])) - 1
log_norm = lambda x: log(x + 1)
KEYS = ["close", "high", "low", "open", "volumefrom", "volumeto"]
MARKETS = ['Bitfinex', 'Coinbase', 'Poloniex', 'Gemini', 'Kraken', 'BitTrex', 'HitBTC', 'Cexio', 'Quoine', 'Exmo']

def normalize_dataframe(markets_df):
    norm_df = markets_df.copy()
    norm_df.loc[:, pd.IndexSlice[:, ["close", "high", "low", "open"]]] = price_normalize(norm_df.loc[:, pd.IndexSlice[:, ["close", "high", "low", "open"]]])
    norm_df.loc[:, pd.IndexSlice[:, ["volumefrom", "volumeto"]]] = log_norm(norm_df.loc[:, pd.IndexSlice[:, ["volumefrom", "volumeto"]]])
    return norm_df


def load_data(databases):
    markets_dataframe = OrderedDict()

    for market in databases:
        print(market)
        market_df = pd.read_csv(path.join(BASE_DIR, "ETH", "csvs", "{}_minute.csv".format(market)), index_col=0)
        assert market_df.shape[1] == len(KEYS)
        markets_dataframe[market] = market_df

    markets_dataframe = pd.concat(markets_dataframe.values(), axis=1, keys=markets_dataframe.keys())
    markets_dataframe = markets_dataframe.fillna(method='ffill')
    assert markets_dataframe.isnull().any().any() == False

    return markets_dataframe

def compute_top_correlated_for_market(markets_df, top_k):
    """
    for each time series return the k most correlated in the same sector
    :param markets_df: dataframe with all the data of all the markets
    :param top_k: top k to return
    :return:
    """
    ret = {}
    for market in MARKETS:
        print(market)
        markets_to_compare = filter(lambda x: x != market, MARKETS)
        market_df = markets_df.loc[:, pd.IndexSlice[market, "close"]]
        corr_coef = [(market_to_compare, abs(market_df["close"].corr(market_df[market_to_compare]["close"]))) for market_to_compare in markets_to_compare]
        sorted_corr_coef = list(map(lambda x:x[0], sorted(corr_coef, key=lambda corr_tuple: corr_tuple[1], reverse=True)))
        ret[market] = sorted_corr_coef[:top_k]

    return ret

def transfor_meta_info(meta_infos, inverse=False):
    if inverse:
        data = [value[0] for key, value in sorted(meta_infos.inverse.items(), key=lambda x: x[0])]
    else:
        data = [value for key, value in sorted(meta_infos.d.items(), key=lambda x: x[0])]
    meta_one_hot, meta_encoder = one_hot_conversion(data)
    return meta_one_hot, meta_encoder




def generate_embedding(markets_df, seq_len=20):
    """
    generate the embeddings of the different timeseries:
    1) generate ids
    2) normalize timeseries
    3) generate pytorch embedding
    :param markets_df: Pandas dataframe containing the stock information of all the tickers
    :param market_correlation: top k correlated tickers
    :param features: features to extract from all the data
    :param ts_lenght: length of each example (timesereis is year long->extract many examples)
    :return:
    """
    num_exp = (markets_df.shape[0] - 1) // seq_len
    features_len = len(KEYS)

    # generate markets ids
    market_to_id = bidict()
    market_id_to_exp_id = MyBidict()

    for market in MARKETS:
        market_to_id[market] = len(market_to_id)



    # generate torch embedding
    input_embeddings = torch.FloatTensor(num_exp, seq_len, features_len).zero_()
    target_embeddings = torch.FloatTensor(num_exp, seq_len, 1).zero_()
    neighbor_embeddings = torch.FloatTensor(num_exp, len(MARKETS) - 1, seq_len, features_len).zero_()


    cb_df = markets_df["Coinbase"]
    idx = cb_df.iloc[:(num_exp * seq_len)].index.values
    t_idx = cb_df.iloc[1:(num_exp * seq_len) + 1].index.values

    cb_ebmedding = torch.from_numpy(cb_df.loc[idx].values).float()
    input_embeddings = cb_ebmedding.view(num_exp, seq_len, -1)
    target_embeddings = torch.from_numpy(cb_df.loc[t_idx]["close"].values).float().view(num_exp, seq_len, 1)

    n_embeddings = torch.FloatTensor(top_k, num_exp * seq_len, features_len).zero_()
    for m_idx, market in enumerate(filter(lambda x: x != "Coinbase", MARKETS)):
        n_market_idx = market_to_id[market]
        n_market_df = markets_df[market]
        n_serie_attribute = torch.from_numpy(n_market_df.loc[idx].values).float()
        n_embeddings[m_idx] = n_serie_attribute

    neighbor_embeddings = torch.stack(torch.split(n_embeddings, seq_len, dim=1), dim=0)
    market_id_to_exp_id[market_to_id["Coinbase"]] = range(num_exp)
    return input_embeddings, target_embeddings, neighbor_embeddings, market_to_id, market_id_to_exp_id

def split_training_test_dataset(market_ids, market_id_to_exp_id, e_t_size=650):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each stock
    :return:
    """

    test_sample = random.sample(market_id_to_exp_id[1], e_t_size)
    for s_idx in test_sample:
        market_id_to_exp_id[1].remove(s_idx)

    eval_sample = random.sample(market_id_to_exp_id[1], e_t_size)
    for s_idx in eval_sample:
        market_id_to_exp_id[1].remove(s_idx)

    train_sample = market_id_to_exp_id[1]
    return train_sample, eval_sample, test_sample


if __name__ == "__main__":
    markets_dataframe = load_data(MARKETS)
    norm_markets_dataframe = normalize_dataframe(markets_dataframe)

    # pickle.dump(norm_markets_dataframe, open(path.join(BASE_DIR, "stock", "norm_dataframe.bin"), "wb"))
    # pickle.dump(compute_top_correlated_for_market(sector_info.inverse, norm_markets_dataframe, 4), open(path.join(BASE_DIR, "stock", "neighbors.bin"), "wb"))

    norm_markets_dataframe = pickle.load(open(path.join(BASE_DIR, "stock", "norm_dataframe.bin"), "rb"))
    input_embeddings, target_embeddings, neighbor_embeddings, market_to_id, market_id_to_exp_id = generate_embedding(norm_markets_dataframe)

    pickle.dump(input_embeddings, open(path.join(BASE_DIR, "stock", "input_embeddings.bin"), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "stock", "target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "stock", "neighbor_embeddings.bin"), "wb"))
    pickle.dump((market_to_id), open(path.join(BASE_DIR, "stock", "market_to_id.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(sorted(market_to_id.inv.keys()), market_id_to_exp_id)
    pickle.dump(train_dataset, open(path.join(BASE_DIR, "stock", "train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "stock", "eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "stock", "test_dataset.bin"), "wb"))



