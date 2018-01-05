from datasets.stock.utils import fetch_quandl_data, MyBidict, BASE_DIR, API_KEY
from os import walk, path
import pandas as pd
import pickle

constitute_info = pd.read_csv(path.join(BASE_DIR, "constituents.csv"))
meta_info = MyBidict()



START = "2017-01-03"
END = "2017-12-01"

for idx, symbol, name, sector in constitute_info.itertuples():
    try:
        data = fetch_quandl_data(symbol, START, END, API_KEY)
        ts_idx, series = next(data.iterrows())
        assert START == data.axes[0][0].strftime('%Y-%m-%d'), "start time error"
        assert END == data.axes[0][-1].strftime('%Y-%m-%d'), "end time error"


        data.to_csv(path.join(BASE_DIR, "csvs", "{}.csv".format(symbol)))
        meta_info[symbol] = sector.lower()
    except Exception as e:
        print(e)
        print(name, sector, symbol)

    if path.isfile(path.join(BASE_DIR, "csvs", "{}.csv".format(symbol))):
        meta_info[symbol] = sector.lower()
print(len(meta_info))
pickle.dump(meta_info, open(path.join(BASE_DIR, "meta_info.bin"), "wb"))




