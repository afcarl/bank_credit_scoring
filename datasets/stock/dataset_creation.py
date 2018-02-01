import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict
import pandas as pd
from collections import OrderedDict
from bidict import bidict
import torch
import random
import datetime as dt

normalize = lambda x: (x.div(x.iloc[0])) - 1
KEYS = ["Open", "High", "Low", "Close", "Volume"]

def load_data(databases, start=dt.datetime(2017, 11, 1, 0, 0), end=dt.datetime(2017, 11, 29, 0, 0)):
    meta_infos = MyBidict()
    symbols_dataframe = OrderedDict()

    for database in databases:
        meta_info = pickle.load(open(path.join(BASE_DIR, "stock", database, "meta_info.bin"), "rb"))
        for symbol in meta_info.keys():
            meta_infos[symbol] = meta_info[symbol]
            symbol_df = pd.read_csv(path.join(BASE_DIR, "stock", database, "csvs", "{}.csv".format(symbol)))
            symbol_df = symbol_df.set_index(pd.DatetimeIndex(symbol_df['Date']))
            symbol_df = symbol_df[start:end][KEYS]
            symbols_dataframe[symbol] = symbol_df
    symbols_dataframe = pd.concat(symbols_dataframe.values(), axis=1, keys=symbols_dataframe.keys())
    symbols_dataframe = symbols_dataframe.fillna(method='ffill')
    assert symbols_dataframe.shape[0] == 21

    return meta_infos, symbols_dataframe

def compute_top_correlated_for_symbol(meta_info_inv, symbols_df, top_k):
    """
    for each time series return the k most correlated in the same sector
    :param meta_info_inv: meta_information (symbol <-> sector)
    :param symbols_df: dataframe with all the data of all the symbols
    :param top_k: top k to return
    :return:
    """
    adj_normalized_df = symbols_df.loc[:, pd.IndexSlice[:, "Close"]]
    ret = {}
    for sector, symbols in meta_info_inv.items():
        for symbol in symbols:
            symbols_to_compare = filter(lambda x: x != symbol, symbols)
            symbol_df = adj_normalized_df[symbol]
            corr_coef = [(symbol_to_compare, abs(symbol_df["Close"].corr(adj_normalized_df[symbol_to_compare]["Close"]))) for
                         symbol_to_compare in symbols_to_compare]
            sorted_corr_coef = list(map(lambda x:x[0], sorted(corr_coef, key=lambda corr_tuple: corr_tuple[1], reverse=True)))
            ret[symbol] = sorted_corr_coef[:top_k]

    return ret



def generate_embedding(symbols_df, symbol_correlation, top_k):
    """
    generate the embeddings of the different timeseries:
    1) generate ids
    2) normalize timeseries
    3) generate pytorch embedding
    :param symbols_df: Pandas dataframe containing the stock information of all the tickers
    :param symbol_correlation: top k correlated tickers
    :param features: features to extract from all the data
    :param ts_lenght: length of each example (timesereis is year long->extract many examples)
    :return:
    """

    ts_lenght = symbols_df.shape[0] - 1
    features_len = len(KEYS)

    # generate symbols ids
    symbol_to_id = bidict()
    for symbol in symbol_correlation.keys():
        symbol_to_id[symbol] = len(symbol_to_id) + 1

    #generate torch embedding
    input_embeddings = torch.FloatTensor(len(symbol_to_id) + 1, ts_lenght, features_len).zero_()
    target_embeddings = torch.FloatTensor(len(symbol_to_id) + 1, ts_lenght, 1).zero_()
    neighbor_embeddings = torch.FloatTensor(len(symbol_to_id) + 1, top_k, ts_lenght, features_len).zero_()

    for symbol, symbol_id in sorted(symbol_to_id.items(), key=lambda x: x[1]):
        symbol_df = symbols_df[symbol]
        input_embeddings[symbol_id] = torch.from_numpy(symbol_df.iloc[:-1].values).float()
        target_embeddings[symbol_id] = torch.from_numpy(symbol_df.iloc[1:]["Close"].values).float()

        n_embeddings = torch.FloatTensor(top_k, ts_lenght, features_len).zero_()
        for idx, n_symbol in enumerate(symbol_correlation[symbol]):
            n_symbol_df = symbols_df[n_symbol]
            n_embeddings[idx] = torch.from_numpy(n_symbol_df.iloc[:-1].values).float()

        neighbor_embeddings[symbol_id] = n_embeddings

    return input_embeddings, target_embeddings, neighbor_embeddings, symbol_to_id

def split_training_test_dataset(stock_ids, e_t_size=700):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each stock
    :return:
    """

    test_sample = random.sample(stock_ids, e_t_size)
    for s_idx in test_sample:
        stock_ids.remove(s_idx)

    eval_sample = random.sample(stock_ids, e_t_size)
    for s_idx in eval_sample:
        stock_ids.remove(s_idx)

    return stock_ids, eval_sample, test_sample



DATABASES = ["india", "tokyo", "usa"]

if __name__ == "__main__":
    # meta_infos, symbols_dataframe = load_data(DATABASES)
    # norm_symbols_dataframe = normalize(symbols_dataframe)
    #
    # pickle.dump(norm_symbols_dataframe, open(path.join(BASE_DIR, "stock", "norm_dataframe.bin"), "wb"))
    # pickle.dump(compute_top_correlated_for_symbol(meta_infos.inverse, norm_symbols_dataframe, 4), open(path.join(BASE_DIR, "stock", "neighbors.bin"), "wb"))
    norm_symbols_dataframe = pickle.load(open(path.join(BASE_DIR, "stock", "norm_dataframe.bin"), "rb"))
    symbol_correlation = pickle.load(open(path.join(BASE_DIR, "stock", "neighbors.bin"), "rb"))
    input_embeddings, target_embeddings, neighbor_embeddings, symbol_to_id = generate_embedding(norm_symbols_dataframe, symbol_correlation, 4)

    pickle.dump(input_embeddings, open(path.join(BASE_DIR, "stock", "input_embeddings.bin"), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "stock", "target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "stock", "neighbor_embeddings.bin"), "wb"))
    pickle.dump((symbol_to_id), open(path.join(BASE_DIR, "stock", "symbol_to_id.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(sorted(symbol_to_id.inv.keys()))
    pickle.dump(train_dataset, open(path.join(BASE_DIR, "stock", "train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "stock", "eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "stock", "test_dataset.bin"), "wb"))



