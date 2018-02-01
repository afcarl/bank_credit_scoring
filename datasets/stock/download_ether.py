import pandas as pd
from datasets.utils import fetch_data_by_exchange
from os.path import join as path_join
import time as T
import datetime as DT
from helper import ensure_dir


grey = .6, .6, .6
TOP_N = 10
# define a pair
FSYM = "ETH"
TSYM = "USD"
MARKETS = ['Bitfinex', 'Coinbase', 'Poloniex', 'Gemini', 'Kraken', 'BitTrex', 'HitBTC', 'Cexio', 'Quoine', 'Exmo']
DATE_START = DT.datetime.strptime("{} 08:00:00".format(DT.datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d %H:%M:%S')
DATE_END = DATE_START - DT.timedelta(days=7)
LIMIT = 2000



def get_data(container, time):
    df, time_to = fetch_data_by_exchange(FSYM, TSYM, market, time, time_frame="minute")
    container.append(df)
    return df.shape[0], time_to

for market in MARKETS:
    print("{}".format(market), end="")
    dfs = []

    num_row, time = get_data(dfs, T.mktime(DATE_START.timetuple()))
    while num_row > LIMIT:
        num_row, time = get_data(dfs, time)

    data = pd.concat(dfs).sort_index().drop_duplicates("time")
    data.to_csv(ensure_dir(path_join("./data", "{}_minute.csv".format(market))))
    print("\tdownloaded")