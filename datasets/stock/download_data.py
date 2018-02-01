from datasets.utils import fetch_quandl_data, MyBidict, BASE_DIR, API_KEY
from os import path
import pandas as pd
import pickle

DATABASES = dict(
    use="WIKI",
    tokyo="TSE",
    india="NSE"
)

START = "2017-01-04"
END = "2017-12-01"


if __name__ == "__main__":
    database = 'india'
    constitute_info = pd.read_csv(path.join(BASE_DIR, database, "constituents.csv"))
    meta_info = MyBidict()

    for idx, symbol, name, sector in constitute_info.itertuples():
        try:
            data = fetch_quandl_data(symbol, START, END, API_KEY, database=DATABASES[database])
            ts_idx, series = next(data.iterrows())

            assert START == data.axes[0][0].strftime('%Y-%m-%d'), "start time error"
            assert END == data.axes[0][-1].strftime('%Y-%m-%d'), "end time error"
            if database == "india":
                data = data.rename(index=str, columns={"Total Trade Quantity": "Volume"})

            data.to_csv(path.join(BASE_DIR, database, "csvs", "{}.csv".format(symbol)))
            meta_info[symbol] = sector.lower()
        except Exception as e:
            print(e)
            print(name, sector, symbol)

        if path.isfile(path.join(BASE_DIR, database, "csvs", "{}.csv".format(symbol))):
            meta_info[symbol] = sector.lower()
    print(len(meta_info))
    pickle.dump(meta_info, open(path.join(BASE_DIR, database, "meta_info.bin"), "wb"))




