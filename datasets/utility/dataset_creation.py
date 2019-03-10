import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict, SiteInfo, one_hot_conversion
import pandas as pd
from collections import OrderedDict
from bidict import bidict
import torch
from helper import ensure_dir
import re
import geopy.distance
from numpy import array, arange, exp, log, nan

TRAIN = [6, 8, 25, 29, 44, 45, 55, 78, 9, 12, 13, 41, 88, 99, 100, 101, 103, 109, 111, 116, 136, 281, 285, 304, 339, 341, 363, 366, 391, 399, 690, 716, 765, 648, 654, 673, 674, 697, 703, 718, 731, 737, 742, 887, 767, 808, 32, 42, 14, 137, 236, 22, 56,  31, 36]
EVAL = [10, 49, 41, 30, 144, 153, 186, 197, 213, 88, 214, 400, 401, 404, 427, 454, 455, 472, 761, 744, 745, 766, 384, 755]
# TEST = [21, 51, 65, 217, 218, 224, 228, 259, 270, 275, 92, 474, 475, 478, 484, 492, 496, 512, 472, 832, 771, 786, 805, 386]
TEST = [10, 21, 30, 41, 49, 51, 65, 88, 92, 144, 153, 186, 197, 213, 214, 217, 218, 224, 228, 259, 270, 275, 384, 386, 400, 401, 404, 427, 454, 455, 472, 474, 475, 478, 484, 492, 496, 512, 744, 745, 755, 761, 766, 771, 786, 805, 832]
KEYS = ["Open", "High", "Low", "Close", "Volume"]
remove_space = lambda x: re.sub('\s', '', x)

softplus = lambda x: log(1 + exp(x))
start_dif = lambda x: x.div(x.iloc[0]) - 1
softlog = lambda x: log(x + 1)
identity = lambda x: x

def setup_norm(type="softlog"):
    if type == "start_dif":
        def normalize(dataframe):
            norm_df = dataframe.copy()
            norm_df.loc[:, pd.IndexSlice[:, ["value"]]] = start_dif(norm_df.loc[:, pd.IndexSlice[:, ["value"]]])
            return norm_df, dataframe.loc[:, pd.IndexSlice[:, ["value"]]].iloc[0]
    elif type == "softlog":
        def normalize(dataframe):
            norm_df = dataframe.copy()
            norm_df.loc[:, pd.IndexSlice[:, ["value", "estimated"]]] = softlog(norm_df.loc[:, pd.IndexSlice[:, ["value", "estimated"]]])
            return norm_df
    elif type == "softplus":
        def normalize(dataframe):
            norm_df = dataframe.copy()
            norm_df.loc[:, pd.IndexSlice[:, ["value", "estimated"]]] = softplus(norm_df.loc[:, pd.IndexSlice[:, ["value", "estimated"]]])
            return norm_df
    else:
        def normalize(dataframe):
            return identity(dataframe)
    return normalize


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
    sectors_one_hot, sector_label_encoder, sector_onehot_encoder = one_hot_conversion(secotor_values)
    tzs_one_hot, tz_label_encoder, tz_onehot_encoder = one_hot_conversion(time_zone_values)



    for (key, value), sector_one_hot, tz_one_hot in zip(sites_attribute.items(), sectors_one_hot, tzs_one_hot):
        example = [log(value.sq_ft), value.lat, value.lng]
        example.extend(sector_one_hot.tolist())
        example.extend(tz_one_hot.tolist())
        ret[value.site_id] = example

    return ret, (sector_label_encoder, sector_onehot_encoder), (tz_label_encoder, tz_onehot_encoder)

def load_data(site_infos, resample_interval="180T", norm_type="softplus"):
    """
    load the data in a pandas datafra
    1) read the attribute
    2) remove anomalies
    3) resample the timeseries in interval of 3 hours
    4) normalize the price for sqrt meter
    5) concatenate and trucante nan
    6) apply some sort of normalization according specification
    7) create day in week feature
    8) create hour in day features
    :param site_infos: info of each site
    :param resample_interval: resample interval
    :param norm_type: type of normalization
    :return:
    """
    sites_dataframe = OrderedDict()
    norm_fn = setup_norm(norm_type)

    for site_id in site_infos.keys():
        site_df = pd.read_csv(path.join(BASE_DIR, "utility", "csvs", "{}.csv".format(site_id)))
        site_df = site_df.set_index(pd.DatetimeIndex(site_df['dttm_utc']))

        # check presence of anomaly, in case remove them
        anomaly_idx = site_df["anomaly"].notnull()
        if anomaly_idx.any():
            site_df = site_df[anomaly_idx == False]
        site_df = site_df.drop("anomaly", axis=1)
        # resample
        site_df = site_df[["value", "estimated"]].resample(resample_interval).sum()
        # sqt normalization
        site_df["value"] /= sites_info[site_id].sq_ft
        sites_dataframe[site_id] = site_df

    sites_dataframe = pd.concat(sites_dataframe.values(), axis=1, keys=sites_dataframe.keys())
    # remove nan timeseries
    sites_dataframe = sites_dataframe.dropna()

    # normalize df
    if norm_type == "start_dif":
        sites_normalized_dataframe, start_values = norm_fn(sites_dataframe)
    elif norm_type == "softlog" or norm_type == "softplus":
        sites_normalized_dataframe = norm_fn(sites_dataframe)
    else:
        sites_normalized_dataframe = sites_dataframe

    # extract day and hours from index
    idx = sites_normalized_dataframe.index
    days, time = (idx.strftime("%A"), idx.strftime("%H"))


    # convert to categorycal
    days_onehot, days_label_encoder, days_onehot_encoder = one_hot_conversion(days)
    days_onehot = pd.DataFrame(days_onehot, index=idx, columns=days_label_encoder.classes_)

    if resample_interval[-1] == "T":
        times_onehot, times_label_encoder, times_onehot_encoder = one_hot_conversion(time)
        times_onehot = pd.DataFrame(times_onehot, index=idx, columns=times_label_encoder.classes_)

    if norm_type == "start_dif" and resample_interval[-1] == "T":
        return sites_normalized_dataframe, start_values, days_onehot, times_onehot
    elif norm_type == "softlog" and resample_interval[-1] == "T":
        return sites_normalized_dataframe, days_onehot, times_onehot
    elif norm_type == "start_dif" and resample_interval[-1] == "D":
        return sites_normalized_dataframe, start_values, days_onehot
    elif norm_type == "softlog" and resample_interval[-1] == "D":
        return sites_normalized_dataframe, days_onehot
    else:
        return sites_normalized_dataframe, days_onehot, times_onehot

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



def generate_embedding(sites_df, sites_attribute, sites_correlation, days_onehot, tz_onehot=None, seq_len=10, features_len=35, top_k=4):
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
    site_to_id = bidict()
    site_id_to_exp_idx = MyBidict()
    num_exp = (sites_df.shape[0] - 1) // seq_len

    # format sites attribute
    sites_attribute, sector_encoder, tz_encoder = convert_attribute(sites_attribute)


    for site in sorted(sites_attribute.keys()):
        site_to_id[site] = len(site_to_id) + 1

    # resample by each day
    input_embeddings = torch.FloatTensor(num_exp * (len(site_to_id) + 1), seq_len, features_len).zero_()
    target_embeddings = torch.FloatTensor(num_exp * (len(site_to_id) + 1), seq_len, 1).zero_()
    neighbor_embeddings = torch.FloatTensor(num_exp * (len(site_to_id) + 1), top_k,  seq_len, features_len).zero_()
    edge_type = torch.ones(num_exp * (len(site_to_id) + 1), top_k, seq_len, 1)
    neigh_mask = torch.zeros(num_exp * (len(site_to_id) + 1), top_k)

    for site, site_id in site_to_id.items():
        site_df = sites_df.loc[:, pd.IndexSlice[site]]

        idx = site_df.iloc[:(num_exp * seq_len)].index.values
        t_idx = site_df.iloc[1:(num_exp * seq_len) + 1].index.values

        # concate time series
        if tz_onehot is not None:
            site_df = pd.concat([site_df, days_onehot, tz_onehot], axis=1)
        else:
            site_df = pd.concat([site_df, days_onehot], axis=1)

        # extract the needed attribute
        att_site = torch.FloatTensor(sites_attribute[site]).unsqueeze(0).expand(idx.shape[0], -1)

        # extract datapoints base on idx
        in_embeddig = torch.from_numpy(site_df.loc[idx].values).float()
        ta_embedding = torch.from_numpy(site_df.loc[t_idx]["value"].values).float()



        # concat att and ts embeddings
        tf_embedding = torch.cat((in_embeddig, att_site), dim=1)

        # split the timseries
        input_embeddings[site_id*num_exp:(site_id + 1) * num_exp] = tf_embedding.view(num_exp, seq_len, -1)
        target_embeddings[site_id*num_exp:(site_id + 1) * num_exp] = ta_embedding.view(num_exp, seq_len, 1)
        # extract neighbors ts
        n_embeddings = torch.FloatTensor(top_k, num_exp * seq_len, features_len).zero_()
        for n_idx, n_site in enumerate(sites_correlation[site]):
            # extract neighbors timeseries
            if tz_onehot is not None:
                n_site_df = torch.from_numpy(
                    pd.concat([sites_df.loc[idx, pd.IndexSlice[n_site]], days_onehot.loc[idx], tz_onehot.loc[idx]],
                              axis=1).values).float()
            else:
                n_site_df = torch.from_numpy(
                    pd.concat([site_df.loc[idx, pd.IndexSlice[n_site]], days_onehot.loc[idx]], axis=1).values).float()

            n_att_site = torch.FloatTensor(sites_attribute[n_site]).unsqueeze(0).expand(idx.shape[0], -1)

            # generate neighbor embedding for the current day
            n_embeddings[n_idx] = torch.cat((n_site_df, n_att_site), dim=1)

        neighbor_embeddings[site_id * num_exp:(site_id + 1) * num_exp] = torch.stack(torch.split(n_embeddings, seq_len, dim=1), dim=0)
        site_id_to_exp_idx[site_id] = list(range(site_id * num_exp, (site_id + 1) * num_exp))

    return input_embeddings, target_embeddings, neighbor_embeddings, edge_type, neigh_mask.byte(), site_to_id, site_id_to_exp_idx

def split_training_test_dataset(site_to_idx, site_to_exp_idx):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each site
    :param id_to_exp_id: example_id for each site
    :param e_t_size: dimentions of the split
    :return:
    """

    test_dataset = []
    eval_dataset = []
    train_dataset = []

    for site_id in TRAIN:
        train_dataset.extend(site_to_exp_idx.d[site_to_idx[site_id]])

    for site_id in EVAL:
        eval_dataset.extend(site_to_exp_idx.d[site_to_idx[site_id]])

    for site_id in TEST:
        test_dataset.extend(site_to_exp_idx.d[site_to_idx[site_id]])

    print("train len: {}\neval len: {}\ntest len: {}".format(len(train_dataset), len(eval_dataset), len(test_dataset)))

    return train_dataset, eval_dataset, test_dataset



if __name__ == "__main__":
    meta_infos, sites_info = read_costituents()
    # sites_normalized_dataframe, days_onehot, tz_onehot = load_data(sites_info)
    #
    # pickle.dump(sites_normalized_dataframe, open(path.join(BASE_DIR, "utility", "temp", "norm_dataframe.bin"), "wb"))
    # pickle.dump(days_onehot, open(path.join(BASE_DIR, "utility", "temp", "days_onehot.bin"), "wb"))
    # pickle.dump(compute_top_correlated(sites_info, 4), open(path.join(BASE_DIR, "utility", "temp", "neighbors.bin"), "wb"))
    # pickle.dump(tz_onehot, open(path.join(BASE_DIR, "utility", "temp", "tz_onehot.bin"), "wb"))
    # pickle.dump(start_values, open(path.join(BASE_DIR, "utility", "temp", "start_values.bin"), "wb"))



    sites_normalized_dataframe = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "norm_dataframe.bin"), "rb"))
    days_onehot = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "days_onehot.bin"), "rb"))
    sites_correlation = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "neighbors.bin"), "rb"))
    tz_onehot = pickle.load(open(path.join(BASE_DIR, "utility", "temp", "tz_onehot.bin"), "rb"))


    input_embeddings, target_embeddings, neighbor_embeddings, edge_types, neigh_mask, site_to_idx, site_to_exp_idx = generate_embedding(sites_normalized_dataframe,
                                                                                                                sites_info,
                                                                                                                sites_correlation,
                                                                                                                days_onehot,
                                                                                                                tz_onehot,
                                                                                                                seq_len=16)

    torch.save(input_embeddings, ensure_dir(path.join(BASE_DIR, "utility", "utility_input_embeddings.pt")))
    torch.save(target_embeddings, ensure_dir(path.join(BASE_DIR, "utility", "target_embeddings.pt")))
    torch.save(neighbor_embeddings, ensure_dir(path.join(BASE_DIR, "utility", "neighbor_embeddings.pt")))
    torch.save(edge_types, ensure_dir(path.join(BASE_DIR, "utility", "edge_type.pt")))
    torch.save(neigh_mask, ensure_dir(path.join(BASE_DIR, "utility", "mask_neighbor.pt")))
    torch.save(site_to_idx, ensure_dir(path.join(BASE_DIR, "utility", "site_to_idx.pt")))
    torch.save(site_to_exp_idx, ensure_dir(path.join(BASE_DIR, "utility", "site_to_exp_idx.pt")))

    site_to_idx = torch.load(path.join(BASE_DIR, "utility", "site_to_idx.pt"))
    site_to_exp_idx = torch.load(path.join(BASE_DIR, "utility", "site_to_exp_idx.pt"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(site_to_idx, site_to_exp_idx)

    torch.save(train_dataset, ensure_dir(path.join(BASE_DIR, "utility", "train_dataset.pt")))
    torch.save(eval_dataset, ensure_dir(path.join(BASE_DIR, "utility", "eval_dataset.pt")))
    torch.save(test_dataset, ensure_dir(path.join(BASE_DIR, "utility", "test_dataset.pt")))



