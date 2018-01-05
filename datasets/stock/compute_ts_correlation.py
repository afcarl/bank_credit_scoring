import pickle
from os import path
from datasets.stock.utils import BASE_DIR
import pandas as pd
from collections import OrderedDict

normalize = lambda x: (x.div(x.loc[0])) - 1

def load_data():
    meta_info = pickle.load(open(path.join(BASE_DIR, "meta_info.bin"), "rb"))
    symbols_dataframe = OrderedDict()

    for symbol in meta_info.keys():
        symbols_dataframe[symbol] = pd.read_csv(path.join(BASE_DIR, "csvs", "{}.csv".format(symbol)))

    symbols_dataframe = pd.concat(symbols_dataframe.values(), axis=1, keys=symbols_dataframe.keys())
    return meta_info, symbols_dataframe.dropna(axis=0, how='any')

def compute_top_correlated_for_symbol(meta_info_inv, symbols_df, top_k):
    """
    for each time series return the k most correlated in the same sector
    :param meta_info_inv: meta_information (symbol <-> sector)
    :param symbols_df: dataframe with all the data of all the symbols
    :param top_k: top k to return
    :return:
    """
    adj_normalized_df = normalize(symbols_df.loc[:, pd.IndexSlice[:, "Adj. Close"]])
    ret = {}
    for sector, symbols in meta_info_inv.items():
        for symbol in symbols:
            symbols_to_compare = filter(lambda x: x != symbol, symbols)
            symbol_df = adj_normalized_df[symbol]
            corr_coef = [(symbol_to_compare, abs(symbol_df["Adj. Close"].corr(adj_normalized_df[symbol_to_compare]["Adj. Close"]))) for
                         symbol_to_compare in symbols_to_compare]
            sorted_corr_coef = list(map(lambda x:x[0], sorted(corr_coef, key=lambda corr_tuple: corr_tuple[1], reverse=True)))
            ret[symbol] = sorted_corr_coef[:top_k]

    return ret







if __name__ == "__main__":
    meta_info, symbols_dataframe = load_data()
    # pickle.dump(compute_top_correlated_for_symbol(meta_info.inverse, symbols_dataframe, 4), open(path.join(BASE_DIR, "neighbors.bin"), "wb"))
