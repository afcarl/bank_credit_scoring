import pickle
import helper
import os.path as path
import numpy as np
import pandas as pd


BASE_DIR = path.join("..", "..", "data", "customers")


def merge_infos():
    accordato_max_dic_1 = pickle.load(open(path.join(BASE_DIR, "temp", "accordato_max_dic_1.bin"), "rb"))
    accordato_max_dic_2 = pickle.load(open(path.join(BASE_DIR, "temp", "accordato_max_dic_2.bin"), "rb"))
    print("read accordato1 {}".format(len(accordato_max_dic_1)))
    print("read accordato2 {}".format(len(accordato_max_dic_2)))
    customers_analytics = {**accordato_max_dic_1, **accordato_max_dic_2}
    print("read merged {}".format(len(customers_analytics)))
    print("number of keys: {}".format(len(customers_analytics.keys())))

    # attribute = ['date_ref', 'value1', 'value2']
    # df_list = []
    # for k, v in customers_analytics.items():
    #     try:
    #         v.columns = pd.MultiIndex.from_product([[k], attribute], names=['id', 'attribute'])
    #         df_list.append(v)
    #     except:
    #         print(k)
    #         print(v)

    customers_analytics = pd.concat(customers_analytics, axis=1)
    print("concatenated")
    customers_analytics.columns = customers_analytics.columns.rename(['id', 'attribute'])
    print(customers_analytics.columns.get_level_values('id').unique())

    customers_analytics.update(customers_analytics.fillna(method='bfill').fillna(method='ffill'))
    customers_analytics.update(customers_analytics.fillna(-1))
    nan_customers = customers_analytics.isnull().any(axis=1, level="id").any()
    print(len(nan_customers))
    customers_analytics.to_msgpack(path.join(BASE_DIR, "temp", "customers_accordato.msg"))


def extract_risk_info():
    customers_analytics = pd.read_msgpack(path.join(BASE_DIR, "temp", "customers_accordato.msg"))
    customers_data = pd.read_msgpack(path.join(BASE_DIR, "temp", "customers_risk_time_frame_null_df_final.msg"))

    print(customers_data.columns.get_level_values('attribute').unique())

    for id in customers_analytics.columns.get_level_values('id').unique():
        risk_info = customers_data.loc[customers_analytics.index, pd.IndexSlice[
            id, ["pre_notching", "val_scoring_risk", "class_scoring_risk", "class_scoring_pre"]]]
        customers_analytics = pd.concat([customers_analytics, risk_info], axis=1)

    customers_analytics.to_msgpack(path.join(BASE_DIR, "temp", "customers_accordato_risk.msg"))
    return customers_analytics

if __name__ == "__main__":
    customers_analytics = extract_risk_info()








