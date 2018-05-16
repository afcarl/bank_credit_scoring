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
            node_data = read_station_data(node)
            node_data = resample_dataframe(node_data)
            assert not np.isnan(node_data.values).any()
            nodes_data[node] = node_data

        neighbors_data = []
        for neighbor_id, distance in G.neighbors(node):
            if neighbor_id in nodes_data:
                neighbor_data = nodes_data[neighbor_id]
            else:
                neighbor_data = read_station_data(neighbor_id)
                neighbor_data = resample_dataframe(neighbor_data)
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

def resample_dataframe(station_dataframe, resample_interval="10T"):
    """
    resample a dataframe
    :param station_dataframe:
    :return:
    """

    def round(t, freq):
        freq = pd.tseries.frequencies.to_offset(freq)
        return pd.Timestamp((t.value // freq.delta.value) * freq.delta.value)

    station_dataframe = station_dataframe.groupby(partial(round, freq=resample_interval)).agg(
        {"Flow (Veh/5 Minutes)": "sum", "Speed (mph)": "mean", "# Lane Points": "mean", "% Observed": "mean"})


    station_dataframe["Flow (Veh/5 Minutes)"] = np.log((station_dataframe["Flow (Veh/5 Minutes)"]/10) +1)
    station_dataframe["Speed (mph)"] = np.log(station_dataframe["Speed (mph)"] + 1)
    station_dataframe["% Observed"] = station_dataframe["% Observed"] / 100
    return station_dataframe

def fix_station_data(stations, ref_station_data):

    for station_idx, station_id in enumerate(stations.index):
        station_data = read_station_data(station_id)

        print(station_idx, station_id)

        if "Speed (mph)" not in station_data.columns:
            station_data["Speed (mph)"] = pd.Series(np.zeros(station_data.index.size), index=station_data.index)
            station_data = station_data[["Flow (Veh/5 Minutes)", "Speed (mph)", "# Lane Points", "% Observed"]]

        if station_data.shape[0] != ref_station_data.shape[0]:
            station_data = station_data.reindex_like(ref_station_data, method='ffill')


        if pd.isnull(station_data).any().any():
            print(station_id)
            raise Exception("bad stations")

        station_data = resample_dataframe(station_data)

        station_data = station_data.apply(pd.to_numeric)
        station_data.to_csv(path.join(BASE_DIR, "pems", "fix", "{}.csv".format(station_id)), date_format="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    stations = read_stations()

    station_id = 400000
    station_data = read_station_data(station_id)
    fix_station_data(stations, station_data)
