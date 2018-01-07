import pickle
from os import path
from datasets.stock.utils import BASE_DIR
import pandas as pd
from collections import OrderedDict, namedtuple
from bidict import bidict
import torch
import random
normalize = lambda x: (x.div(x.loc[0])) - 1



def load_data():
    meta_info = pickle.load(open(path.join(BASE_DIR, "meta_info.bin"), "rb"))
    symbols_dataframe = OrderedDict()

    for symbol in meta_info.keys():
        symbols_dataframe[symbol] = pd.read_csv(path.join(BASE_DIR, "csvs", "{}.csv".format(symbol)))

    symbols_dataframe = pd.concat(symbols_dataframe.values(), axis=1, keys=symbols_dataframe.keys())
    return meta_info, symbols_dataframe.dropna(axis=0, how='any')

def compute_top_correlated_for_symbol(meta_info_inv, symbols_df, top_k):
    """
    for each time series return the k most correlated in the same sector
    :param meta_info_inv: meta_information (symbol <-> sector)
    :param symbols_df: dataframe with all the data of all the symbols
    :param top_k: top k to return
    :return:
    """
    adj_normalized_df = normalize(symbols_df.loc[:, pd.IndexSlice[:, "Adj. Close"]])
    ret = {}
    for sector, symbols in meta_info_inv.items():
        for symbol in symbols:
            symbols_to_compare = filter(lambda x: x != symbol, symbols)
            symbol_df = adj_normalized_df[symbol]
            corr_coef = [(symbol_to_compare, abs(symbol_df["Adj. Close"].corr(adj_normalized_df[symbol_to_compare]["Adj. Close"]))) for
                         symbol_to_compare in symbols_to_compare]
            sorted_corr_coef = list(map(lambda x:x[0], sorted(corr_coef, key=lambda corr_tuple: corr_tuple[1], reverse=True)))
            ret[symbol] = sorted_corr_coef[:top_k]

    return ret



def generate_embedding(symbols_df, symbol_correlation,
                       features=["Ex-Dividend", "Split Ratio", "Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"],
                       ts_lenght=12):
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
    exp_for_stock = int(symbols_df.shape[0] / ts_lenght)
    max_seq_len = exp_for_stock * ts_lenght

    # generate symbols ids
    symbol_to_id = bidict()
    for symbol in symbol_correlation.keys():
        symbol_to_id[symbol] = len(symbol_to_id) + 1

    #keep only needed timeseries
    symbols_df = symbols_df.loc[:, pd.IndexSlice[:, features]]

    # normalize all the timeseries
    norm_symbols_df = normalize(symbols_df)
    norm_symbols_df.loc[:, pd.IndexSlice[:, "Ex-Dividend"]] = symbols_df.loc[:, pd.IndexSlice[:, "Ex-Dividend"]]
    assert not norm_symbols_df.isnull().values.any(), "nan value present in the normalized values"



    #generate torch embedding
    input_embeddings = torch.FloatTensor((len(symbol_to_id) + 1) * exp_for_stock, ts_lenght, len(features)).zero_()
    target_embeddings = torch.FloatTensor((len(symbol_to_id) + 1) * exp_for_stock, ts_lenght, 1).zero_()
    neighbor_embeddings = torch.FloatTensor((len(symbol_to_id) + 1) * exp_for_stock, len(symbol_correlation["A"]), ts_lenght, len(features)).zero_()

    for symbol, symbol_id in sorted(symbol_to_id.items(), key=lambda x: x[1]):
        symbol_df = norm_symbols_df[symbol]
        input_embedding = torch.from_numpy(symbol_df.iloc[0:max_seq_len, :].values).float()
        target_embedding = torch.from_numpy(symbol_df.iloc[1:max_seq_len+1]["Adj. Close"].values).float()
        input_embeddings[symbol_id * exp_for_stock:(symbol_id + 1) * exp_for_stock] = input_embedding.view(exp_for_stock, ts_lenght, -1)
        target_embeddings[symbol_id * exp_for_stock:(symbol_id + 1) * exp_for_stock] = target_embedding.view(exp_for_stock, ts_lenght, -1)

        n_embeddings = torch.FloatTensor(exp_for_stock, len(symbol_correlation["A"]), ts_lenght, len(features)).zero_()
        for idx, n_symbol in enumerate(symbol_correlation[symbol]):
            n_symbol_df = norm_symbols_df[n_symbol]
            n_embedding = torch.from_numpy(n_symbol_df.iloc[0:max_seq_len, :].values).float()
            n_embeddings[:, idx] = n_embedding.view(exp_for_stock, ts_lenght, -1)

        neighbor_embeddings[symbol_id * exp_for_stock:(symbol_id + 1) * exp_for_stock] = n_embeddings

    return input_embeddings, target_embeddings, neighbor_embeddings, symbol_to_id, exp_for_stock

def split_training_test_dataset(stock_ids, exp_for_stock, e_t_size=80):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each stock
    :return:
    """

    test_sample = random.sample(stock_ids, e_t_size)
    test_dataset = []
    for s_idx in test_sample:
        test_dataset.extend(list(range(s_idx*exp_for_stock, (s_idx+1)*exp_for_stock)))
        stock_ids.remove(s_idx)

    eval_sample = random.sample(stock_ids, e_t_size)
    eval_dataset = []
    for s_idx in eval_sample:
        eval_dataset.extend(list(range(s_idx * exp_for_stock, (s_idx + 1) * exp_for_stock)))
        stock_ids.remove(s_idx)

    train_dataset = []
    for s_idx in stock_ids:
        train_dataset.extend(list(range(s_idx, s_idx + exp_for_stock)))

    return stock_ids, eval_sample, test_sample




if __name__ == "__main__":
    meta_info, symbols_dataframe = load_data()
    # pickle.dump(compute_top_correlated_for_symbol(meta_info.inverse, symbols_dataframe, 4), open(path.join(BASE_DIR, "neighbors.bin"), "wb"))
    symbol_correlation = pickle.load(open(path.join(BASE_DIR, "neighbors.bin"), "rb"))
    input_embeddings, target_embeddings, neighbor_embeddings, symbol_to_id, exp_for_stock = generate_embedding(symbols_dataframe, symbol_correlation)

    pickle.dump(input_embeddings, open(path.join(BASE_DIR, "input_embeddings.bin"), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "neighbor_embeddings.bin"), "wb"))
    pickle.dump((symbol_to_id, exp_for_stock), open(path.join(BASE_DIR, "symbol_to_id_and_exp_for_stock.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(sorted(symbol_to_id.inv.keys()), exp_for_stock)
    pickle.dump(train_dataset, open(path.join(BASE_DIR, "train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "test_dataset.bin"), "wb"))



