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
        if path.isfile(path.join(BASE_DIR, "pems", "fix", "{}.csv".format(station_id))):
            continue

        if station_data["Flow (Veh/5 Minutes)"].dtype == 'object':
            station_data["Flow (Veh/5 Minutes)"] = station_data["Flow (Veh/5 Minutes)"].map(lambda x: float(x.replace(",", "")))

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
        station_data.to_csv(ensure_dir(path.join(BASE_DIR, "pems", "fix", "{}.csv".format(station_id))), date_format="%Y-%m-%d %H:%M:%S")


def combine_stations_info(data_dir):
    stations = pd.read_csv(path.join(data_dir, "stations.csv"), index_col="ID")
    station_lat_lng = pd.read_csv(path.join(data_dir, "lat_lng.csv"), index_col="ID")
    print("df size: {}".format(stations.shape))

    stations = pd.concat([stations, station_lat_lng], axis=1, join='inner')
    stations.to_csv(path.join(data_dir, "stations_comp.csv"))
    print("df size: {}".format(stations.shape))

    to_remove = []
    for idx, id in enumerate(stations.index):
        if not path.isfile(path.join(data_dir, "stations", "{}.csv".format(id))):
            print("missing {}-{}".format(idx, id))
            to_remove.append(id)

    if len(to_remove) > 0:
        print(to_remove)
        stations = stations.drop(to_remove, axis="index")
        print("df size: {}".format(stations.shape))

    stations.to_csv(path.join(data_dir, "stations_comp.csv"))
    return stations

if __name__ == "__main__":
    stations = combine_stations_info(path.join(BASE_DIR, "pems"))

    ref_station_id = 400000
    ref_station_data = read_station_data(ref_station_id)
    fix_station_data(stations, ref_station_data)
