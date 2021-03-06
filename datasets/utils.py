from bs4 import BeautifulSoup
from datetime import datetime
import requests
import json
import numpy as np
import pandas as pd
from collections import namedtuple
from os.path import join
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


ExganceInfo = namedtuple("ExganceInfo", ["market", "volume"])
SiteInfo = namedtuple("SiteInfo", ["site_id", "sector", "sq_ft", "lat", "lng", "time_zone", "tz_offset"])

API_KEY = "zTEsWpGga_5eqG6YCkRS"
BASE_DIR = join("..", "..", "data")




def fetch_data_by_exchange(fswym, tsym, exchange, time_to, time_frame="hour"):
    """
    Get hourly data from cryptocompare
    :param fswym: From Symbol
    :param tsym: To Symbols
    :param exchange: Name of exchange
    :return:
    """
    url = "https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym={}&limit=2000&e={}&toTs={}".\
        format(time_frame, fswym, tsym, exchange, time_to)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    dic = json.loads(soup.prettify())
    next_time_to = dic["TimeFrom"] - 1
    date = np.array([datetime.utcfromtimestamp(int(info["time"])).strftime('%Y-%m-%d %H:%M:%S') for info in dic["Data"]])
    data = pd.DataFrame.from_dict(dic["Data"])
    data.index = pd.to_datetime(date)
    return data.astype(np.float32), next_time_to


def fetch_quandl_data(symbol, oldest_available_date, newest_availabele_date, api, database="WIKI"):
    quandl.ApiConfig.api_key = api
    try:
        data = quandl.get("{}/{}".format(database, symbol), collapse="daily", start_date=oldest_available_date, end_date=newest_availabele_date)
        return data
    except ValueError as e:
        print(e)



class MyBidict(object):
    def __init__(self):
        self.d = {}
        self.inverse = {}

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        self.d[key] = value
        if type(value) == list:
            for val in value:
                self.inverse.setdefault(val, []).append(key)
        else:
            self.inverse.setdefault(value, []).append(key)

    def __len__(self):
        return len(self.d)

    def __contains__(self, item):
        return self.d.__contains__(item)

    def keys(self):
        return sorted(self.d.keys())


def one_hot_conversion(values, label_encoder=None, one_hot_encoder=None):
    if type(values) == list:
        values = np.array(values)

    if label_encoder == None:
        label_encoder = LabelEncoder()
        values_encoded = label_encoder.fit_transform(values)
    else:
        values_encoded = label_encoder.transform(values)
    values_encoded = values_encoded.reshape(len(values_encoded), 1)

    if one_hot_encoder == None:
        one_hot_encoder = OneHotEncoder(sparse=False)
        one_hot_encoded = one_hot_encoder.fit_transform(values_encoded)
    else:
        one_hot_encoded = one_hot_encoder.transform(values_encoded)

    return one_hot_encoded, label_encoder, one_hot_encoder
