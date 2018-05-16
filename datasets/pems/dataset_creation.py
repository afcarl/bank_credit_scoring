import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict, SiteInfo, one_hot_conversion
import pandas as pd
from collections import OrderedDict
from bidict import bidict
import torch
import networkx as nx
# from geopy.distance import geodesic
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

def read_station_data(station_id, lanes):
    """
    read the csv dataframe of a given staton
    :param station_id: 
    :return: 
    """
    station_data = pd.read_csv(path.join(BASE_DIR, "pems", "fix", "{}.csv".format(station_id)))
    station_data['Unnamed: 0'] = pd.to_datetime(station_data['Unnamed: 0'], format="%Y-%m-%d %H:%M:%S")
    station_data = station_data.set_index('Unnamed: 0')
    station_data["# Lane Points"] = lanes
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
        data["Flow (Veh/5 Minutes)"] = data["Flow (Veh/5 Minutes)"].sum()
        data["Speed (mph)"] = data["Speed (mph)"].mean()
        data["# Lane Points"] = data["# Lane Points"].mean()
        data["% Observed"] = data["% Observed"].mean()
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



def generate_embedding(stations, G, top_k=6):
    station_id_to_idx = bidict()
    station_id_to_exp_idx = MyBidict()

    # station_id = 408134
    station_id = 400000
    station_data = read_station_data(station_id, stations.loc[station_id, "Lanes"])
    days_groups = get_days_datapoints(station_data)

    # one hot encoders
    _, day_label_encoder, day_one_hot_encoder = generate_one_hot_encoding(station_data.index.day)
    _, hour_label_encoder, hour_one_hot_encoder = generate_one_hot_encoding(station_data.index.hour)
    _, minutes_label_encoder, minutes_one_hot_encoder = generate_one_hot_encoding(station_data.index.minute)

    nodes = list(filter(lambda x: type(x) == int, G.nodes))
    num_exp = day_one_hot_encoder.active_features_.size
    seq_len = days_groups[0].size - 1
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
            node_data = read_station_data(node, stations.loc[station_id, "Lanes"])
            assert not np.isnan(node_data.values).any()
            nodes_data[node] = node_data

        neighbors_data = []
        for neighbor_id, distance in G.neighbors(node):
            if neighbor_id in nodes_data:
                neighbor_data = nodes_data[neighbor_id]
            else:
                neighbor_data = read_station_data(neighbor_id, stations.loc[station_id, "Lanes"])
                assert not np.isnan(neighbor_data.values).any()
                nodes_data[neighbor_id] = neighbor_data
            neighbors_data.append((neighbor_id, neighbor_data))

        station_id_to_idx[node] = node_idx

        # node embedding
        for day_idx, day_timestep in enumerate(days_groups):
            day_one_hot, _, _ = one_hot_conversion(day_timestep.day, day_label_encoder, day_one_hot_encoder)
            hour_one_hot, _, _ = one_hot_conversion(day_timestep.hour, hour_label_encoder, hour_one_hot_encoder)
            minute_one_hot, _, _ = one_hot_conversion(day_timestep.minute, minutes_label_encoder, minutes_one_hot_encoder)

            node_data_value = np.concatenate([node_data.loc[day_timestep].values, day_one_hot, hour_one_hot, minute_one_hot], axis=1)
            input_embeddings[((node_idx * num_exp) + day_idx):((node_idx * num_exp) + day_idx + 1)] = torch.from_numpy(node_data_value[:-1])
            target_embeddings[((node_idx*num_exp)+day_idx):((node_idx*num_exp)+day_idx+1)] = torch.from_numpy(node_data_value[1:, 0])

            # neighbor embedding
            for neighbor_idx, (neighbor_id, neighbor_data) in enumerate(neighbors_data):
                try:
                    neighbor_data_value = np.concatenate([neighbor_data.loc[day_timestep].values, day_one_hot, hour_one_hot, minute_one_hot], axis=1)
                    neighbor_embeddings[((node_idx*num_exp)+day_idx), neighbor_idx] = torch.from_numpy(neighbor_data_value[:-1])
                except Exception as e:
                    print(neighbor_idx, neighbor_id, day_idx)
                    print(e)
                    raise e

        station_id_to_exp_idx[node] = list(range(node_idx * num_exp, (node_idx + 1) * num_exp))

        if node_idx % 10 == 0:
            print(node_idx)

    return input_embeddings, target_embeddings, neighbor_embeddings, edge_type, neigh_mask, station_id_to_idx, station_id_to_exp_idx

def split_training_test_dataset(site_to_idx, site_to_exp_idx, train_stations, eval_stations, test_stations):
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

    for _id in train_stations:
        train_dataset.extend(site_to_exp_idx.d[_id])

    for _id in eval_stations:
        eval_dataset.extend(site_to_exp_idx.d[_id])

    for _id in test_stations:
        test_dataset.extend(site_to_exp_idx.d[_id])

    print("train len: {}\neval len: {}\ntest len: {}".format(len(train_dataset), len(eval_dataset), len(test_dataset)))

    return train_dataset, eval_dataset, test_dataset



if __name__ == "__main__":
    stations = read_stations()
    # G, stations_distances = compute_graph(stations)

    G = torch.load(path.join(BASE_DIR, "pems", "temp", "graph.pt"))

    input_embeddings, target_embeddings, neighbor_embeddings, edge_type, neigh_mask, station_id_to_idx, station_id_to_exp_idx = generate_embedding(stations, G)

    torch.save(input_embeddings, ensure_dir(path.join(BASE_DIR, "pems", "input_embeddings.pt")))
    torch.save(target_embeddings, ensure_dir(path.join(BASE_DIR, "pems", "target_embeddings.pt")))
    torch.save(neighbor_embeddings, ensure_dir(path.join(BASE_DIR, "pems", "neighbor_embeddings.pt")))
    torch.save(edge_type, ensure_dir(path.join(BASE_DIR, "pems", "edge_type.pt")))
    torch.save(neigh_mask, ensure_dir(path.join(BASE_DIR, "pems", "mask_neighbor.pt")))
    torch.save(station_id_to_idx, ensure_dir(path.join(BASE_DIR, "pems", "station_id_to_idx.pt")))
    torch.save(station_id_to_exp_idx, ensure_dir(path.join(BASE_DIR, "pems", "station_id_to_exp_idx.pt")))

    station_id_to_idx = torch.load(path.join(BASE_DIR, "pems", "station_id_to_idx.pt"))
    station_id_to_exp_idx = torch.load(path.join(BASE_DIR, "pems", "station_id_to_exp_idx.pt"))


    stations_id = sorted(stations.index.values)
    train_stations = np.random.choice(stations_id, 2500, replace=False)
    for station_id in train_stations:
        stations_id.remove(station_id)

    eval_stations = np.random.choice(stations_id, 600, replace=False)
    for station_id in eval_stations:
        stations_id.remove(station_id)





    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(station_id_to_idx, station_id_to_exp_idx,
                                                                            train_stations, eval_stations, stations_id)

    torch.save(train_dataset, ensure_dir(path.join(BASE_DIR, "pems", "train_dataset.pt")))
    torch.save(eval_dataset, ensure_dir(path.join(BASE_DIR, "pems", "eval_dataset.pt")))
    torch.save(test_dataset, ensure_dir(path.join(BASE_DIR, "pems", "test_dataset.pt")))



