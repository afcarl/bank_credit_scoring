import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict, SiteInfo
import pandas as pd
from collections import OrderedDict, Counter
from bidict import bidict
import torch
import random
import datetime as dt
import re
import numpy as np
import geopy.distance

normalize = lambda x: (x.div(x.iloc[0])) - 1
KEYS = ["Open", "High", "Low", "Close", "Volume"]
remove_space = lambda x: re.sub('\s', '', x)

def one_hot(labels, C):
    one_hot = [0]*C
    one_hot[labels] = 1
    return one_hot


def read_costituents():
    site_info = pd.read_csv(path.join(BASE_DIR, "utility", "constituents.csv"))
    meta_infos = MyBidict()
    site_infos = OrderedDict()
    for id, site_id, industry, sector, sq_ft, lat, lng, time_zone, tz_offset in site_info.itertuples():
        sector = "{}_{}".format(remove_space(industry), remove_space(sector))
        meta_infos[site_id] = sector
        site_infos[site_id] = SiteInfo(site_id, sector, sq_ft, lat, lng, time_zone, tz_offset)

    return meta_infos, site_infos


def convert_attribute(symbols_attribute):
    sector_to_idx = bidict()
    tz_to_idx = bidict()

    ret = OrderedDict()

    for value in symbols_attribute.values():
        if value.sector not in sector_to_idx:
            sector_to_idx[value.sector] = len(sector_to_idx)

        if value.time_zone not in tz_to_idx:
            tz_to_idx[value.time_zone] = len(tz_to_idx)


    for key, value in symbols_attribute.items():
        example = [value.sq_ft, value.lat, value.lng]
        example.extend(one_hot(sector_to_idx[value.sector], len(sector_to_idx)))
        example.extend(one_hot(tz_to_idx[value.time_zone], len(tz_to_idx)))
        ret[key] = example

    return ret, sector_to_idx, tz_to_idx

def load_data(site_infos, resample_interval="60T"):
    symbols_dataframe = OrderedDict()

    for site_id in site_infos.keys():
        site_df = pd.read_csv(path.join(BASE_DIR, "utility", "csvs", "{}.csv".format(site_id)))
        site_df = site_df.set_index(pd.DatetimeIndex(site_df['dttm_utc']))

        # check presence of anomaly, in case remove them
        anomaly_idx = site_df["anomaly"].notnull()
        if anomaly_idx.any():
            site_df = site_df[anomaly_idx == False]
        site_df = site_df.drop("anomaly", axis=1)
        symbols_dataframe[site_id] = site_df

    symbols_dataframe = pd.concat(symbols_dataframe.values(), axis=1, keys=symbols_dataframe.keys())

    # align timeseries
    symbols_dataframe = symbols_dataframe.dropna()
    # resample timeseries
    symbols_dataframe = symbols_dataframe.loc[:, pd.IndexSlice[:, ["value", "estimated"]]].resample(resample_interval).sum()
    return symbols_dataframe

def compute_top_correlated_for_symbol(symbols_attribute, top_k):
    """
    for each time series return the k most correlated in the same sector
    :param meta_info_inv: meta_information (symbol <-> sector)
    :param symbols_df: dataframe with all the data of all the symbols
    :param top_k: top k to return
    :return:
    """

    ret = {}
    symbols = symbols_attribute.keys()
    for symbol in symbols:
        symbols_to_compare = filter(lambda x: x != symbol, symbols)
        s_coords = symbols_attribute[symbol].lat, symbols_attribute[symbol].lng
        corr_coef = [(c_symbol, geopy.distance.vincenty(s_coords, (symbols_attribute[c_symbol].lat, symbols_attribute[c_symbol].lng)).km) for c_symbol in symbols_to_compare]
        sorted_corr_coef = list(map(lambda x:x[0], sorted(corr_coef, key=lambda corr_tuple: corr_tuple[1])))
        ret[symbol] = sorted_corr_coef[:top_k]

    return ret



def generate_embedding(symbols_df, symbols_attribute, symbol_correlation, example_len=10):
    """
    generate the embeddings of the different timeseries:
    1) convert the attribute in a feature vector
    1) generate symbol-ids
    2) subsample the timeseries to daly interval
    3) generate pytorch embedding
    :param symbols_df: Pandas dataframe containing the stock information of all the tickers
    :param symbols_attribute: symbols attribute
    :param symbol_correlation: top k correlated tickers
    :param example_len: example length
    :return:
    """
    # format symbols attribute
    symbols_attribute, sector_to_idx, tz_to_idx = convert_attribute(symbols_attribute)

    # generate symbols ids
    symbol_to_id = bidict()
    for symbol in symbol_correlation.keys():
        symbol_to_id[symbol] = len(symbol_to_id) + 1

    id_to_exp_id = {}

    # resample by each day
    group_symbols_df = symbols_df.resample('D')
    input_embeddings = []
    target_embeddings = []
    neighbor_embeddings = []

    for symbol, symbol_id in sorted(symbol_to_id.items(), key=lambda x: x[1]):
        # save the current number of example
        start_len = len(input_embeddings)
        # extract the needed timeseries
        symbol_df = symbols_df.loc[:, pd.IndexSlice[symbol, ["value", "estimated"]]]
        # extract the needed attribute
        att_symbol = torch.FloatTensor(symbols_attribute[symbol])
        for day, day_idx in group_symbols_df.indices.items():
            # take the timestemps for a given day
            tf_symbol = symbol_df.iloc[day_idx]
            # check that it is long enough
            if tf_symbol.shape[0] > example_len + 1:
                tf_symbol = torch.FloatTensor(tf_symbol.values)
                # create node embedding
                tf_embedding = torch.cat((att_symbol.unsqueeze(0).expand(tf_symbol.size(0), -1), tf_symbol), dim=1)
                # split node embedding in different example of fixed length
                num_example = ((tf_embedding.size(0) - 1) // example_len)
                # generate node and target embedding for current day
                day_input_embedding = tf_embedding[:num_example*example_len]
                day_target_embedding = tf_embedding[1:(num_example*example_len) + 1, -2]

                # generate the different example for the current day
                day_input_embedding = torch.split(day_input_embedding, example_len, dim=0)
                day_target_embedding = torch.split(day_target_embedding, example_len, dim=0)

                input_embeddings.extend(day_input_embedding)
                target_embeddings.extend(day_target_embedding)

                n_embeddings = []
                for idx, n_symbol in enumerate(symbol_correlation[symbol]):
                    # extract neighbors timeseries
                    n_symbol_df = symbols_df.loc[:, pd.IndexSlice[n_symbol, ["value", "estimated"]]]
                    n_tf_symbol = torch.FloatTensor(n_symbol_df.iloc[day_idx].values)
                    # extract neighbor attribute
                    n_att_symbol = torch.FloatTensor(symbols_attribute[n_symbol])

                    # generate neighbor embedding for the current day
                    day_n_embedding = torch.cat((n_att_symbol.unsqueeze(0).expand(tf_symbol.size(0), -1), n_tf_symbol), dim=1)
                    day_n_embedding = day_n_embedding[:num_example*example_len].view(num_example, example_len, -1)
                    n_embeddings.append(day_n_embedding)

                neighbor_embeddings.extend(torch.split(torch.stack(n_embeddings).transpose(0, 1), 1, dim=0))

        # save the index of the examples of the same node
        id_to_exp_id[symbol_id] = torch.LongTensor(list(range(start_len, len(input_embeddings))))

    return torch.stack(input_embeddings), torch.stack(target_embeddings), torch.cat(neighbor_embeddings), symbol_to_id, id_to_exp_id

def split_training_test_dataset(stock_ids, id_to_exp_id, e_t_size=25):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each symbol
    :param id_to_exp_id: example_id for each symbol
    :param e_t_size: dimentions of the split
    :return:
    """

    test_sample = random.sample(stock_ids, e_t_size)
    test_dataset = []
    for s_idx in test_sample:
        test_dataset.extend(id_to_exp_id[s_idx])
        stock_ids.remove(s_idx)

    eval_sample = random.sample(stock_ids, e_t_size)
    eval_dataset = []
    for s_idx in eval_sample:
        eval_dataset.extend(id_to_exp_id[s_idx])
        stock_ids.remove(s_idx)

    train_dataset = []
    for s_idx in stock_ids:
        train_dataset.extend(id_to_exp_id[s_idx])
    return train_dataset, eval_dataset, test_dataset



if __name__ == "__main__":
    meta_infos, site_infos = read_costituents()
    symbols_dataframe = load_data(site_infos)

    # symbols_dataframe.loc[:, pd.IndexSlice[:, "value"]] = normalize(symbols_dataframe.loc[:, pd.IndexSlice[:, "value"]])

    pickle.dump(symbols_dataframe, open(path.join(BASE_DIR, "norm_dataframe.bin"), "wb"))
    pickle.dump(compute_top_correlated_for_symbol(site_infos, 4), open(path.join(BASE_DIR, "neighbors.bin"), "wb"))


    symbols_dataframe = pickle.load(open(path.join(BASE_DIR, "norm_dataframe.bin"), "rb"))
    symbol_correlation = pickle.load(open(path.join(BASE_DIR, "neighbors.bin"), "rb"))
    input_embeddings, target_embeddings, neighbor_embeddings, symbol_to_id, id_to_exp_id = generate_embedding(symbols_dataframe, site_infos, symbol_correlation)

    pickle.dump(input_embeddings, open(path.join(BASE_DIR, "input_embeddings.bin"), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "neighbor_embeddings.bin"), "wb"))
    pickle.dump((symbol_to_id), open(path.join(BASE_DIR, "symbol_to_id.bin"), "wb"))
    pickle.dump((id_to_exp_id), open(path.join(BASE_DIR, "id_to_exp_id.bin"), "wb"))


    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(sorted(symbol_to_id.inv.keys()), id_to_exp_id)

    pickle.dump(train_dataset, open(path.join(BASE_DIR, "train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "test_dataset.bin"), "wb"))



