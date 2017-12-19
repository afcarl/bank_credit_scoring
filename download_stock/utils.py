from bs4 import BeautifulSoup
from datetime import datetime
import requests
import json
import numpy as np
import pandas as pd
from collections import namedtuple
import quandl
from helper import ensure_dir

ExganceInfo = namedtuple("ExganceInfo", ["market", "volume"])



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


def fetch_quandl_data(symbol, oldest_available_date, newest_availabele_date, api):
    quandl.ApiConfig.api_key = api
    try:
        data = quandl.get("WIKI/{}".format(symbol), collapse="daily", start_date=oldest_available_date, end_date=newest_availabele_date)
        data.to_csv(ensure_dir("../data/{}.csv".format(symbol)))
    except ValueError as e:
        print(e)


