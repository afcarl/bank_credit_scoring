import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict, SiteInfo, one_hot_conversion
import pandas as pd
from collections import OrderedDict
from bidict import bidict
import torch
import networkx as nx
from geopy.distance import geodesic
import numpy as np
from functools import partial
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helper import ensure_dir

softplus = lambda x: np.log(1 + np.exp(x))
start_dif = lambda x: x.div(x.iloc[0]) - 1
softlog = lambda x: np.log(x + 1)
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


def read_stations():
    """
    read the stations and check the data availability
    :return: 
    """
    stations = pd.read_csv(path.join(BASE_DIR, "pems", "station_comp.csv"), index_col="ID")
    # for idx, id in enumerate(stations.index):
    #     if not path.isfile(path.join(BASE_DIR, "pems", "stations", "{}.csv".format(id))):
    #         print("missing {}-{}".format(idx, id))

    return stations



def compute_graph(stations, top_k=6):
    """
    compute the graph structure of the different stations
    :param stations: 
    :param top_k: 
    :return: 
    """

    stations_distances = OrderedDict()
    G = nx.DiGraph()
    for idx, station_id in enumerate(stations.index):
        stations_cord = stations.loc[station_id, ["Latitude", "Longitude"]].values
        stations_distances[station_id] = OrderedDict()
        for other_station_id in stations.index:
            if other_station_id != station_id:
                other_station_cord = stations.loc[other_station_id, ["Latitude", "Longitude"]].values
                distance = geodesic(stations_cord, other_station_cord).m
                stations_distances[station_id][other_station_id] = distance

        closest_stations = sorted(stations_distances[station_id].items(), key=lambda x: x[1])
        closest_stations = list(filter(lambda x: x[1] > 0, closest_stations))
        for close_station in closest_stations[:top_k]:
            G.add_edge(station_id, close_station)

        print("station-{} complete".format(station_id))
    return G, stations_distances

def read_station_data(station_id):
    """
    read the csv dataframe of a given staton
    :param station_id: 
    :return: 
    """
    station_data = pd.read_csv(path.join(BASE_DIR, "pems", "stations", "{}.csv".format(station_id)))
    station_data["5 Minutes"] = pd.to_datetime(station_data["5 Minutes"], format="%Y-%m-%d %H:%M:%S")
    station_data = station_data.set_index("5 Minutes")
    return station_data

def resample_dataframe(station_dataframe, resample_interval="10T"):
    """
    resample a dataframe
    :param station_dataframe: 
    :return: 
    """

    def round(t, freq):
        freq = pd.tseries.frequencies.to_offset(freq)
        return pd.Timestamp((t.value // freq.delta.value) * freq.delta.value)


    def agg_fn(data):
        if data.name == "Flow (Veh/5 Minutes)":
            data = data.sum()
        else:
            data = data.mean()
        return data

    station_dataframe = station_dataframe.groupby(partial(round, freq=resample_interval)).apply(agg_fn)
    scalar = MinMaxScaler()
    station_dataframe = pd.DataFrame(scalar.fit_transform(station_dataframe), columns=station_dataframe.columns, index=station_dataframe.index)
    return station_dataframe

def get_days_datapoints(station_data):
    """
    get the timestemp of each day
    :param station_data: 
    :return: 
    """
    days_groups = station_data.groupby(station_data.index.day)
    return [v if len(v) > 0 else None for k,v in sorted(days_groups.groups.items(), key=lambda x: x[0])]


def generate_one_hot_encoding(values):
    """
    generate the label and one_hot encoder for time features
    :param values: 
    :return: 
    """
    values_encoded, label_encoder, one_hot_encoder = one_hot_conversion(values)
    return values_encoded, label_encoder, one_hot_encoder



def generate_embedding(G, top_k=6):
    station_id_to_idx = bidict()
    station_id_to_exp_idx = MyBidict()

    station_id = 400000
    station_data = read_station_data(station_id)
    station_data = resample_dataframe(station_data)
    days_groups = get_days_datapoints(station_data)

    # one hot encoders
    _, day_label_encoder, day_one_hot_encoder = generate_one_hot_encoding(station_data.index.day)
    _, hour_label_encoder, hour_one_hot_encoder = generate_one_hot_encoding(station_data.index.hour)
    _, minutes_label_encoder, minutes_one_hot_encoder = generate_one_hot_encoding(station_data.index.minute)

    nodes = list(filter(lambda x: type(x) == int, G.nodes))
    num_exp = day_one_hot_encoder.active_features_.size
    seq_len = days_groups[0].size
    features_len = 4 + day_one_hot_encoder.active_features_.size + hour_one_hot_encoder.active_features_.size + minutes_one_hot_encoder.active_features_.size

    input_embeddings = torch.FloatTensor(num_exp * len(nodes), seq_len, features_len).zero_()
    target_embeddings = torch.FloatTensor(num_exp * len(nodes), seq_len, 1).zero_()
    neighbor_embeddings = torch.FloatTensor(num_exp * len(nodes), top_k, seq_len, features_len).zero_()
    edge_type = torch.ones(num_exp * len(nodes), top_k, seq_len, 1)
    neigh_mask = torch.zeros(num_exp * len(nodes), top_k).byte()

    nodes_data = {}
    for node_idx, node in enumerate(nodes):
        if node in nodes_data:
            node_data = nodes_data[node]
        else:
            node_data = read_station_data(node)
            node_data = resample_dataframe(node_data)
            nodes_data[node] = node_data

        neighbors_data = []
        for neighbor in G.neighbors(node):
            if neighbor in nodes_data:
                neighbor_data = nodes_data[neighbor]
            else:
                neighbor_data = read_station_data(neighbor)
                neighbor_data = resample_dataframe(neighbor_data)
                nodes_data[neighbor] = neighbor_data
            neighbors_data.append(neighbors_data)

        station_id_to_idx[node] = node_idx

    num_exp = (sites_df.shape[0] - 1) // seq_len

    # format sites attribute
    sites_attribute, sector_encoder, tz_encoder = convert_attribute(sites_attribute)


    for site in sorted(sites_attribute.keys()):
        site_to_id[site] = len(site_to_id) + 1

    # resample by each day

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

    return input_embeddings, target_embeddings, neighbor_embeddings, site_to_id, site_id_to_exp_idx

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
    stations = read_stations()
    # G, stations_distances = compute_graph(stations)

    G = torch.load(path.join(BASE_DIR, "pems", "temp", "graph.pt"))

    input_embeddings, target_embeddings, neighbor_embeddings, site_to_idx, site_to_exp_idx = generate_embedding(G)



    # sites_normalized_dataframe, days_onehot, tz_onehot = load_data(sites_info)
    #
    # pickle.dump(sites_normalized_dataframe, open(path.join(BASE_DIR, "utility", "temp", "norm_dataframe.bin"), "wb"))
    # pickle.dump(days_onehot, open(path.join(BASE_DIR, "utility", "temp", "days_onehot.bin"), "wb"))
    # pickle.dump(compute_top_correlated(sites_info, 4), open(path.join(BASE_DIR, "utility", "temp", "neighbors.bin"), "wb"))
    # pickle.dump(tz_onehot, open(path.join(BASE_DIR, "utility", "temp", "tz_onehot.bin"), "wb"))
    # pickle.dump(start_values, open(path.join(BASE_DIR, "utility", "temp", "start_values.bin"), "wb"))



    # input_embeddings, target_embeddings, neighbor_embeddings, site_to_idx, site_to_exp_idx = generate_embedding(sites_normalized_dataframe,
    #                                                                                                             sites_info,
    #                                                                                                             sites_correlation,
    #                                                                                                             days_onehot,
    #                                                                                                             tz_onehot,
    #                                                                                                             seq_len=16)

    # torch.save(input_embeddings, ensure_dir(path.join(BASE_DIR, "utility", "input_embeddings.bin")))
    # torch.save(target_embeddings, ensure_dir(path.join(BASE_DIR, "utility", "target_embeddings.bin")))
    # torch.save(neighbor_embeddings, ensure_dir(path.join(BASE_DIR, "utility", "neighbor_embeddings.bin")))
    # torch.save(site_to_idx, ensure_dir(path.join(BASE_DIR, "utility", "site_to_idx.bin")))
    # torch.save(site_to_exp_idx, ensure_dir(path.join(BASE_DIR, "utility", "site_to_exp_idx.bin")))
    #
    #
    # train_dataset, eval_dataset, test_dataset = split_training_test_dataset(site_to_idx, site_to_exp_idx)
    #
    # torch.save(train_dataset, ensure_dir(path.join(BASE_DIR, "utility", "train_dataset.bin")))
    # torch.save(eval_dataset, ensure_dir(path.join(BASE_DIR, "utility", "eval_dataset.bin")))
    # torch.save(test_dataset, ensure_dir(path.join(BASE_DIR, "utility", "test_dataset.bin")))



