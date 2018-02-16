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
KEYS = ["Open", "High", "Low", "Close", "Volume"]

def normalize_dataframe(symbols_df):
    norm_df = symbols_df.copy()
    norm_df.loc[:, pd.IndexSlice[:, ["Open", "High", "Low", "Close"]]] = price_normalize(norm_df.loc[:, pd.IndexSlice[:, ["Open", "High", "Low", "Close"]]])
    norm_df.loc[:, pd.IndexSlice[:, ["Volume"]]] = log_norm(norm_df.loc[:, pd.IndexSlice[:, ["Volume"]]])
    return norm_df



def load_meta(databases):
    sector_info = MyBidict()
    country_info = MyBidict()

    for database in databases:
        meta_info = pickle.load(open(path.join(BASE_DIR, "stock", database, "meta_info_new.bin"), "rb"))
        country_symbols = []
        for symbol in sorted(meta_info.keys()):
            s_symbol = symbol if type(symbol) == str else str(symbol)
            sector_info[s_symbol] = meta_info[symbol]
            country_symbols.append(s_symbol)
        country_info[database] = country_symbols
    return sector_info, country_info

def load_data(databases, country_info, start=dt.datetime(2017, 10, 3, 0, 0), end=dt.datetime(2017, 11, 29, 0, 0)):
    symbols_dataframe = OrderedDict()

    for database in databases:
        print(database)
        for symbol in country_info.d[database]:
            symbol_df = pd.read_csv(path.join(BASE_DIR, "stock", database, "csvs", "{}.csv".format(symbol)))
            symbol_df = symbol_df.set_index(pd.DatetimeIndex(symbol_df['Date']))
            symbol_df = symbol_df[start:end][KEYS]
            symbols_dataframe[symbol] = symbol_df
    symbols_dataframe = pd.concat(symbols_dataframe.values(), axis=1, keys=symbols_dataframe.keys())
    symbols_dataframe = symbols_dataframe.fillna(method='ffill')
    assert symbols_dataframe.shape[0] == 42
    assert symbols_dataframe.isnull().any().any() == False

    return symbols_dataframe

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
        print(sector)
        for symbol in symbols:
            symbols_to_compare = filter(lambda x: x != symbol, symbols)
            symbol_df = adj_normalized_df[symbol]
            corr_coef = [(symbol_to_compare, abs(symbol_df["Close"].corr(adj_normalized_df[symbol_to_compare]["Close"]))) for
                         symbol_to_compare in symbols_to_compare]
            sorted_corr_coef = list(map(lambda x:x[0], sorted(corr_coef, key=lambda corr_tuple: corr_tuple[1], reverse=True)))
            ret[symbol] = sorted_corr_coef[:top_k]

    return ret

def transfor_meta_info(meta_infos, inverse=False):
    if inverse:
        data = [value[0] for key, value in sorted(meta_infos.inverse.items(), key=lambda x: x[0])]
    else:
        data = [value for key, value in sorted(meta_infos.d.items(), key=lambda x: x[0])]
    meta_one_hot, meta_encoder = one_hot_conversion(data)
    return meta_one_hot, meta_encoder




def generate_embedding(symbols_df, symbol_correlation, sector_info, country_info, top_k, seq_len=10):
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
    sector_one_hot, sector_encoder = transfor_meta_info(sector_info)
    country_one_hot, country_encoder = transfor_meta_info(country_info, inverse=True)

    num_exp = (symbols_df.shape[0] - 1) // seq_len
    features_len = len(KEYS) + sector_one_hot.shape[1] + country_one_hot.shape[1]

    # generate symbols ids
    symbol_to_id = bidict()
    symbol_id_to_exp_id = MyBidict()

    for symbol in sorted(symbol_correlation.keys()):
        symbol_to_id[symbol] = len(symbol_to_id) + 1



    # generate torch embedding
    input_embeddings = torch.FloatTensor(num_exp * (len(symbol_to_id) + 1), seq_len, features_len).zero_()
    target_embeddings = torch.FloatTensor(num_exp * (len(symbol_to_id) + 1), seq_len, 1).zero_()
    neighbor_embeddings = torch.FloatTensor(num_exp * (len(symbol_to_id) + 1), top_k, seq_len, features_len).zero_()





    for symbol, symbol_id in sorted(symbol_to_id.items(), key=lambda x: x[0]):
        symbol_df = symbols_df[symbol]
        idx = symbol_df.iloc[:(num_exp * seq_len)].index.values
        t_idx = symbol_df.iloc[1:(num_exp * seq_len) + 1].index.values

        serie_attribute = torch.from_numpy(symbol_df.loc[idx].values).float()
        sector_attribute = torch.from_numpy(sector_one_hot[symbol_id - 1]).float().unsqueeze(0).expand(idx.shape[0], -1)
        country_attribute = torch.from_numpy(country_one_hot[symbol_id - 1]).float().unsqueeze(0).expand(idx.shape[0], -1)

        input_embeddings[symbol_id * num_exp:(symbol_id + 1) * num_exp] = torch.cat((serie_attribute, sector_attribute, country_attribute), dim=-1).view(num_exp, seq_len, -1)
        target_embeddings[symbol_id*num_exp:(symbol_id + 1) * num_exp] = torch.from_numpy(symbol_df.loc[t_idx]["Close"].values).float().view(num_exp, seq_len, 1)

        n_embeddings = torch.FloatTensor(top_k, num_exp * seq_len, features_len).zero_()
        for n_idx, n_symbol in enumerate(symbol_correlation[symbol]):
            n_symbol_idx = symbol_to_id[n_symbol]
            n_symbol_df = symbols_df[n_symbol]
            n_serie_attribute = torch.from_numpy(n_symbol_df.loc[idx].values).float()
            n_sector_attribute = torch.from_numpy(sector_one_hot[n_symbol_idx - 1]).float().unsqueeze(0).expand(idx.shape[0], -1)
            n_country_attribute = torch.from_numpy(country_one_hot[n_symbol_idx - 1]).float().unsqueeze(0).expand(idx.shape[0], -1)
            n_embeddings[n_idx] = torch.cat((n_serie_attribute, n_sector_attribute, n_country_attribute), dim=-1)

        neighbor_embeddings[symbol_id*num_exp:(symbol_id + 1) * num_exp] = torch.stack(torch.split(n_embeddings, seq_len, dim=1), dim=0)
        symbol_id_to_exp_id[symbol_id] = list(range(symbol_id*num_exp, (symbol_id + 1) * num_exp))

    return input_embeddings, target_embeddings, neighbor_embeddings, symbol_to_id, symbol_id_to_exp_id

def split_training_test_dataset(stock_ids, symbol_id_to_exp_id, e_t_size=650):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each stock
    :return:
    """

    test_id_sample = random.sample(stock_ids, e_t_size)
    test_sample = []
    for s_idx in test_id_sample:
        test_sample.extend(symbol_id_to_exp_id.d[s_idx])
        stock_ids.remove(s_idx)

    eval_id_sample = random.sample(stock_ids, e_t_size)
    eval_sample = []
    for s_idx in eval_id_sample:
        eval_sample.extend(symbol_id_to_exp_id.d[s_idx])
        stock_ids.remove(s_idx)

    train_sample = []
    for s_idx in stock_ids:
        train_sample.extend(symbol_id_to_exp_id.d[s_idx])

    return train_sample, eval_sample, test_sample



DATABASES = ["india", "tokyo", "usa"]

if __name__ == "__main__":
    sector_info, country_info = load_meta(DATABASES)
    # symbols_dataframe = load_data(DATABASES, country_info)
    # norm_symbols_dataframe = normalize_dataframe(symbols_dataframe)
    #
    # pickle.dump(norm_symbols_dataframe, open(path.join(BASE_DIR, "stock", "norm_dataframe.bin"), "wb"))
    # pickle.dump(compute_top_correlated_for_symbol(sector_info.inverse, norm_symbols_dataframe, 4), open(path.join(BASE_DIR, "stock", "neighbors.bin"), "wb"))





    norm_symbols_dataframe = pickle.load(open(path.join(BASE_DIR, "stock", "norm_dataframe.bin"), "rb"))
    symbol_correlation = pickle.load(open(path.join(BASE_DIR, "stock", "neighbors.bin"), "rb"))
    input_embeddings, target_embeddings, neighbor_embeddings, symbol_to_id, symbol_id_to_exp_id = generate_embedding(norm_symbols_dataframe, symbol_correlation, sector_info, country_info, 4)

    pickle.dump(input_embeddings, open(path.join(BASE_DIR, "stock", "input_embeddings.bin"), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "stock", "target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "stock", "neighbor_embeddings.bin"), "wb"))
    pickle.dump((symbol_to_id), open(path.join(BASE_DIR, "stock", "symbol_to_id.bin"), "wb"))
    pickle.dump((symbol_id_to_exp_id), open(path.join(BASE_DIR, "stock", "symbol_id_to_exp_id.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(sorted(symbol_to_id.inv.keys()), symbol_id_to_exp_id)
    pickle.dump(train_dataset, open(path.join(BASE_DIR, "stock", "train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "stock", "eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "stock", "test_dataset.bin"), "wb"))



