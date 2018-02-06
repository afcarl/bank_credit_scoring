import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict, SiteInfo
import pandas as pd
from collections import OrderedDict
from bidict import bidict
import torch
import random
import datetime as dt
import re
import geopy.distance
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array, arange, exp, log

KEYS = ["Open", "High", "Low", "Close", "Volume"]
remove_space = lambda x: re.sub('\s', '', x)

softplus = lambda x: log(1 + exp(x))
start_dif = lambda x: x.div(x.iloc[0]) - 1

def normalize(dataframe):
    norm_df = dataframe.copy()
    norm_df.loc[:, pd.IndexSlice[:, ["value"]]] = start_dif(norm_df.loc[:, pd.IndexSlice[:, ["value"]]])
    return norm_df, dataframe.loc[:, pd.IndexSlice[:, ["value"]]].iloc[0]

def one_hot_conversion(values):
    encoder = LabelEncoder()
    if type(values) == list:
        values = array(values)

    values_encoded = encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    values_encoded = values_encoded.reshape(len(values_encoded), 1)
    one_hot_encoded = onehot_encoder.fit_transform(values_encoded)

    return one_hot_encoded, encoder


def read_costituents():
    site_info = pd.read_csv(path.join(BASE_DIR, "utility", "constituents.csv"))
    meta_infos = MyBidict()
    site_infos = OrderedDict()
    for id, site_id, industry, sector, sq_ft, lat, lng, time_zone, tz_offset in site_info.itertuples():
        sector = "{}_{}".format(remove_space(industry), remove_space(sector))
        meta_infos[site_id] = sector
        site_infos[site_id] = SiteInfo(site_id, sector, sq_ft, lat, lng, time_zone, tz_offset)

    return meta_infos, site_infos


def convert_attribute(sites_attribute):
    ret = OrderedDict()

    # extract value
    secotor_values = [value.sector for value in sites_attribute.values()]
    time_zone_values = [value.time_zone for value in sites_attribute.values()]

    # one_hot_conversion
    sectors_one_hot, sector_encoder = one_hot_conversion(secotor_values)
    tzs_one_hot, tz_encoder = one_hot_conversion(time_zone_values)



    for (key, value), sector_one_hot, tz_one_hot in zip(sites_attribute.items(), sectors_one_hot, tzs_one_hot):
        example = [value.sq_ft, value.lat, value.lng]
        example.extend(sector_one_hot.tolist())
        example.extend(tz_one_hot.tolist())
        ret[value.site_id] = example

    return ret, sector_encoder, tz_encoder

def load_data(site_infos, resample_interval="60T"):
    sites_dataframe = OrderedDict()

    for site_id in site_infos.keys():
        site_df = pd.read_csv(path.join(BASE_DIR, "utility", "csvs", "{}.csv".format(site_id)))
        site_df = site_df.set_index(pd.DatetimeIndex(site_df['dttm_utc']))

        # check presence of anomaly, in case remove them
        anomaly_idx = site_df["anomaly"].notnull()
        if anomaly_idx.any():
            site_df = site_df[anomaly_idx == False]
        site_df = site_df.drop("anomaly", axis=1)
        sites_dataframe[site_id] = site_df

    sites_dataframe = pd.concat(sites_dataframe.values(), axis=1, keys=sites_dataframe.keys())
    # resample timeseries and remove timeseries
    sites_dataframe = sites_dataframe.loc[:, pd.IndexSlice[:, ["value", "estimated"]]].resample(resample_interval).sum()
    sites_dataframe = sites_dataframe.dropna()

    # normalize df
    sites_normalized_dataframe, start_values = normalize(sites_dataframe)
    # extract day and hours from index
    idx = sites_normalized_dataframe.index
    days, time = (idx.strftime("%A"), idx.strftime("%H"))

    # convert to categorycal
    days_onehot, days_encoder = one_hot_conversion(days)
    times_onehot, times_encoder = one_hot_conversion(time)

    # convert to pandas dataframe
    pd_days_onehot = pd.DataFrame(days_onehot, index=idx, columns=days_encoder.classes_)
    pd_tz_onehot = pd.DataFrame(times_onehot, index=idx, columns=times_encoder.classes_)

    return sites_normalized_dataframe, start_values, pd_days_onehot, pd_tz_onehot

def compute_top_correlated(sites_attribute, top_k):
    """
    for each time series return the k most correlated in the same sector
    :param meta_info_inv: meta_information (symbol <-> sector)
    :param sites_df: dataframe with all the data of all the sites
    :param top_k: top k to return
    :return:
    """

    ret = {}
    sites = sites_attribute.keys()
    for site in sites:
        sites_to_compare = filter(lambda x: x != site, sites)
        s_coords = sites_attribute[site].lat, sites_attribute[site].lng
        corr_coef = [(c_site, geopy.distance.vincenty(s_coords, (sites_attribute[c_site].lat, sites_attribute[c_site].lng)).km) for c_site in sites_to_compare]
        sorted_corr_coef = list(map(lambda x: x[0], sorted(corr_coef, key=lambda corr_tuple: corr_tuple[1])))
        ret[site] = sorted_corr_coef[:top_k]

    return ret



def generate_embedding(sites_normalized_df, days_df, hours_df, sites_attribute, sites_correlation, example_len=10):
    """
    generate the embeddings of the different timeseries:
    1) convert the attribute in a feature vector
    1) generate site-ids
    2) subsample the timeseries to daly interval
    3) generate pytorch embedding
    :param sites_df: Pandas dataframe containing the stock information of all the tickers
    :param sites_attribute: sites attribute
    :param sites_correlation: top k correlated tickers
    :param example_len: example length
    :return:
    """
    site_id_to_idx = bidict()
    site_id_to_exp_idx = {}

    # format sites attribute
    sites_attribute, sector_encoder, tz_encoder = convert_attribute(sites_attribute)

    # resample by each day
    input_embeddings = []
    target_embeddings = []
    neighbor_embeddings = []

    for site in sorted(sites_attribute.keys()):
        start_len = len(input_embeddings)

        # extract the needed timeseries
        site_normalized_df = torch.from_numpy(pd.concat([sites_normalized_df.loc[:, pd.IndexSlice[site, ["value", "estimated"]]], days_df, hours_df], axis=1).values).float()

        # extract the needed attribute
        att_site = torch.FloatTensor(sites_attribute[site])

        # concat att and ts embeddings
        tf_embedding = torch.cat((site_normalized_df, att_site.unsqueeze(0).expand(site_normalized_df.size(0), -1)), dim=1)

        # compute the number of examples
        num_example = ((tf_embedding.size(0) - 1) // example_len)

        # split the timseries
        input_embedding = torch.split(tf_embedding[:num_example * example_len], example_len)
        target_embedding = torch.split(tf_embedding[1:(num_example * example_len) + 1, 0], example_len)
        # extract neighbors ts
        n_embeddings = []
        for n_site in sites_correlation[site]:
            # extract neighbors timeseries
            n_site_df = torch.from_numpy(pd.concat([sites_normalized_df.loc[:, pd.IndexSlice[n_site, ["value", "estimated"]]], days_df, hours_df], axis=1).values).float()
            n_att_site = torch.FloatTensor(sites_attribute[n_site])

            # generate neighbor embedding for the current day
            n_embedding = torch.cat((n_site_df, n_att_site.unsqueeze(0).expand(n_site_df.size(0), -1)), dim=1)
            n_embedding = n_embedding[:num_example * example_len]
            n_embeddings.append(n_embedding)

        input_embeddings.extend(input_embedding)
        target_embeddings.extend(target_embedding)
        neighbor_embeddings.extend(torch.split(torch.stack(n_embeddings), example_len, dim=1))
        site_id_to_exp_idx[site] = list(range(start_len, len(input_embeddings)))
        site_id_to_idx[site] = len(site_id_to_idx)

    return torch.stack(input_embeddings), torch.stack(target_embeddings), torch.stack(neighbor_embeddings).squeeze(), site_id_to_idx, site_id_to_exp_idx

def split_training_test_dataset(site_to_exp_idx, e_t_size=20):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each site
    :param id_to_exp_id: example_id for each site
    :param e_t_size: dimentions of the split
    :return:
    """

    test_sample = random.sample(site_to_exp_idx.keys(), e_t_size)
    test_dataset = []
    for s_idx in test_sample:
        test_dataset.extend(site_to_exp_idx[s_idx])
        site_to_exp_idx.pop(s_idx, None)

    eval_sample = random.sample(site_to_exp_idx.keys(), e_t_size)
    eval_dataset = []
    for s_idx in eval_sample:
        eval_dataset.extend(site_to_exp_idx[s_idx])
        site_to_exp_idx.pop(s_idx, None)

    train_dataset = []
    for s_idx in site_to_exp_idx:
        train_dataset.extend(site_to_exp_idx[s_idx])

    return train_dataset, eval_dataset, test_dataset



if __name__ == "__main__":
    meta_infos, sites_info = read_costituents()
    # sites_normalized_dataframe, start_values, days_onehot, tz_onehot = load_data(sites_info)
    #
    # pickle.dump(sites_normalized_dataframe, open(path.join(BASE_DIR, "utility", "temp", "norm_dataframe.bin"), "wb"))
    # pickle.dump(start_values, open(path.join(BASE_DIR, "utility", "temp", "start_values.bin"), "wb"))
    # pickle.dump(days_onehot, open(path.join(BASE_DIR, "utility", "temp", "days_onehot.bin"), "wb"))
    # pickle.dump(tz_onehot, open(path.join(BASE_DIR, "utility", "temp", "tz_onehot.bin"), "wb"))
    # pickle.dump(compute_top_correlated(sites_info, 4), open(path.join(BASE_DIR, "utility", "temp", "neighbors.bin"), "wb"))

    sites_normalized_dataframe = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "norm_dataframe.bin"), "rb"))
    start_values = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "start_values.bin"), "rb"))
    days_onehot = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "days_onehot.bin"), "rb"))
    tz_onehot = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "tz_onehot.bin"), "rb"))
    sites_correlation = pickle.load(open(path.join(BASE_DIR,"utility", "temp", "neighbors.bin"), "rb"))

    input_embeddings, target_embeddings, neighbor_embeddings, site_to_idx, site_to_exp_idx = generate_embedding(sites_normalized_dataframe,
                                                                                                            days_onehot,
                                                                                                            tz_onehot,
                                                                                                            sites_info,
                                                                                                            sites_correlation)

    pickle.dump(input_embeddings, open(path.join(BASE_DIR, "utility", "input_embeddings.bin"), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "utility", "target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "utility", "neighbor_embeddings.bin"), "wb"))
    pickle.dump((site_to_idx), open(path.join(BASE_DIR, "utility", "site_to_idx.bin"), "wb"))
    pickle.dump((site_to_exp_idx), open(path.join(BASE_DIR, "utility", "site_to_exp_idx.bin"), "wb"))


    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(site_to_exp_idx)

    pickle.dump(train_dataset, open(path.join(BASE_DIR, "utility", "train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "utility", "eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "utility", "test_dataset.bin"), "wb"))



