import mysql.connector
import pandas as pd
import networkx as nx
import pickle
from os import path
from collections import OrderedDict
from bidict import bidict
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from datetime import datetime
import random


STRING_ATTRIBUTE = ['zipcode', 'segmento', 'b_partner', 'cod_uo', 'region', 'country_code', 'customer_kind', 'customer_type',
                    'uncollectable_status', 'ateco', 'sae']
NUMERIC_ATTRIBUTE = ['pre_notching', 'val_scoring_risk', 'val_scoring_pre', 'val_scoring_ai', 'val_scoring_cr', 'val_scoring_bi',
                     'val_scoring_sd', 'class_scoring_risk', 'class_scoring_pre', 'class_scoring_ai', 'class_scoring_cr', 'class_scoring_bi',
                     'class_scoring_sd', 'age']
CLASS_ATTRIBUTE = ['age', 'class_scoring_risk', 'class_scoring_pre', 'class_scoring_ai', 'class_scoring_cr', 'class_scoring_bi', 'class_scoring_sd']
VAL_ATTRIBUTE = ['pre_notching', 'val_scoring_risk', 'val_scoring_pre', 'val_scoring_ai', 'val_scoring_cr', 'val_scoring_bi', 'val_scoring_sd']


BASE_DIR = path.join("..", "..", "data", "customers")


softlog = lambda x: torch.log(x + 2)


def isnan(x):
    return x != x

def normalize_num_attribute(X):
    X = softlog(X)
    assert not isnan(X).any(), "Nan in the numeric attribute"
    assert not X.sum() == float("-inf"), "-Inf in the numeric attribute"
    return X


def __att_tansform__(customer_id, df, attribute, encoder_dict):
    """transform the categorical attribute.
    1) categorical attribute => label
    2) label => one hot vector"""

    label_encoder = encoder_dict[attribute]['label_encoder']
    oh_encoder = encoder_dict[attribute]['one_hot_encoder']
    return oh_encoder.transform(np.expand_dims(label_encoder.transform(df[customer_id, attribute]), 1)).toarray()


def __extract_normalize_attribute_torch__(customer_id, num_attribute_df,  cat_attribute_df, encoder_dict):
    """
    trasfrom the customer dataframe in an appropriate representation pytorch tensor
    :param customer_id:
    :param num_attribute_df:
    :param cat_attribute_df:
    :param encoder_dict:
    :return:
    """
    # convert categorical attribute
    cat_attribute = [torch.FloatTensor(__att_tansform__(customer_id, cat_attribute_df, _att_, encoder_dict))
                     for _att_ in STRING_ATTRIBUTE]

    # convert numerical attribute
    c_attribute = num_attribute_df.loc[:, pd.IndexSlice[customer_id, CLASS_ATTRIBUTE]].values
    v_attribute = num_attribute_df.loc[:, pd.IndexSlice[customer_id, VAL_ATTRIBUTE]].values
    c_attribute[c_attribute <= -1] = -1
    v_attribute[v_attribute <= -1] = -1

    num_attribute = torch.FloatTensor(np.concatenate([c_attribute, v_attribute], axis=1))
    one_hot_attribute = torch.cat(cat_attribute, dim=1)
    try:
        num_attribute = normalize_num_attribute(num_attribute)
        ret_att = torch.cat([num_attribute[:-1], one_hot_attribute[:-1]], dim=1)
        ret_target = num_attribute[1:, 1]

        return ret_att, ret_target
    except Exception as e:
        print("customer id", customer_id)
        print("num_attribute", num_attribute)
        print("one_hot_attribute", one_hot_attribute)
        raise e


def __label_one_hot_category_encoders__(customers_data):
    """
    Compute the label and one-hot encoders for the categorical features
    :return:
    """
    num_cat_features = 0
    cat_encoder_dict = {}
    # create the labels and transformers for the STRING_ATTRIBUTE
    for cat_att in STRING_ATTRIBUTE:
        label_encoder = LabelEncoder()
        one_hot_encoder = OneHotEncoder()
        cat_attribute_label = label_encoder.fit_transform(customers_data.loc[:, pd.IndexSlice[:, cat_att]].values.flatten())
        one_hot_encoder.fit(np.expand_dims(cat_attribute_label, 1))

        assert label_encoder.classes_.size == one_hot_encoder.active_features_.size
        num_cat_features += label_encoder.classes_.size

        cat_encoder_dict[cat_att] = dict(label_encoder=label_encoder,
                                         one_hot_encoder=one_hot_encoder)
    return num_cat_features, cat_encoder_dict

def __label_one_hot_edge_type_encode__(G):
    edge_types = list(set(v for k,v in nx.get_edge_attributes(G, "rel_type").items()))
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()
    edge_types_label = label_encoder.fit_transform(edge_types)
    one_hot_encoder.fit(edge_types_label.reshape(-1, 1))

    assert label_encoder.classes_.size == one_hot_encoder.active_features_.size
    num_edge_type_features = label_encoder.classes_.size
    print("edge type feature sized: {}".format(num_edge_type_features))

    return num_edge_type_features, dict(label_encoder=label_encoder,
                                        one_hot_encoder=one_hot_encoder)





def extract_dateframe(customers_data, G):
    """
    extract data and compute one_hot_vectors
    :param customers_data:
    :return:
    """
    num_edge_type_features, edge_type_encoder_dict = __label_one_hot_edge_type_encode__(G)

    num_cat_features, cat_encoder_dict = __label_one_hot_category_encoders__(customers_data)

    num_attribute_df = customers_data.loc[:, pd.IndexSlice[:, NUMERIC_ATTRIBUTE]]
    cat_attribute_df = customers_data.loc[:, pd.IndexSlice[:, STRING_ATTRIBUTE]]

    num_attribute_df.to_msgpack(path.join(BASE_DIR, "temp", "num_attribute_df.msg"))
    cat_attribute_df.to_msgpack(path.join(BASE_DIR, "temp", "cat_attribute_df.msg"))
    pickle.dump(cat_encoder_dict, open(path.join(BASE_DIR, "temp", "cat_encoder_dict.bin"), "wb"))


    num_attribute_df = pd.read_msgpack(path.join(BASE_DIR, "temp", "num_attribute_df.msg"))
    cat_attribute_df = pd.read_msgpack(path.join(BASE_DIR, "temp", "cat_attribute_df.msg"))
    cat_encoder_dict = pickle.load(open(path.join(BASE_DIR, "temp", "cat_encoder_dict.bin"), "rb"))

    customer_ids = customers_data.columns.get_level_values("id").unique()
    customer_id_to_idx = bidict()


    seq_len = num_attribute_df.shape[0] - 1
    print(num_cat_features)
    num_feature = num_cat_features + len(NUMERIC_ATTRIBUTE)
    print(num_feature)
    max_neighbors = 10

    customers_embedding = torch.zeros((customer_ids.size, seq_len, num_feature))
    targets_embedding = torch.FloatTensor(customer_ids.size, seq_len, 1).zero_()
    neighbors_embedding = torch.zeros((customer_ids.size, max_neighbors, seq_len, num_feature))
    neighbors_type_embedding = torch.zeros((customer_ids.size, max_neighbors, seq_len, num_edge_type_features))
    neighbors_mask = torch.ones(customer_ids.size, 1).byte()
    custumers_with_out_degree = []
    custumers_without_out_degree = []

    for customer_idx, customer_id in enumerate(customer_ids):
        c_embedding, c_target_embedding = __extract_normalize_attribute_torch__(customer_id, num_attribute_df, cat_attribute_df, cat_encoder_dict)

        customers_embedding[customer_idx] = c_embedding
        targets_embedding[customer_idx] = c_target_embedding

        # extract neighbors att
        num_neighbors = G.out_degree(customer_id)

        assert num_neighbors <= max_neighbors
        for n_idx, (c_id, n_id, rel_type) in enumerate(G.out_edges(customer_id, data="rel_type")):
            assert c_id == customer_id
            n_embedding, _ = __extract_normalize_attribute_torch__(n_id, num_attribute_df, cat_attribute_df, cat_encoder_dict)
            neighbors_embedding[customer_idx, n_idx] = n_embedding
            neighbors_mask[customer_idx] += 1
            rel_one_hot = edge_type_encoder_dict["one_hot_encoder"].transform(
                edge_type_encoder_dict["label_encoder"].transform([rel_type]).reshape(-1, 1))
            neighbors_type_embedding[customer_idx, n_idx] = torch.FloatTensor(np.repeat(rel_one_hot.toarray(), seq_len, axis=0))


        customer_id_to_idx[customer_id] = customer_idx
        if num_neighbors > 0:
            custumers_with_out_degree.append(customer_id)
        else:
            custumers_without_out_degree.append(customer_id)

        if customer_idx % 200 == 0:
            print(customer_idx)

    return customers_embedding, targets_embedding, neighbors_embedding, neighbors_mask, neighbors_type_embedding, customer_id_to_idx, custumers_with_out_degree, custumers_without_out_degree



def split_training_test_dataset(_ids, id_to_idx, e_t_size=2000):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each stock
    :return:
    """

    test_id_dataset = random.sample(_ids, e_t_size)
    test_idx_dataset = []
    for c_id in test_id_dataset:
        test_idx_dataset.append(id_to_idx[c_id])
        _ids.remove(c_id)

    eval_id_dataset = random.sample(_ids, e_t_size)
    eval_idx_dataset = []
    for c_id in eval_id_dataset:
        eval_idx_dataset.append(id_to_idx[c_id])
        _ids.remove(c_id)

    train_idx_dataset = []
    for c_id in _ids:
        train_idx_dataset.append(id_to_idx[c_id])


    return train_idx_dataset, eval_idx_dataset, test_idx_dataset



if __name__ == "__main__":
    customers_data = pd.read_msgpack(path.join(BASE_DIR, "customers_risk_df.msg"))
    G = nx.readwrite.gpickle.read_gpickle(path.join(BASE_DIR, "prune_graph.bin"))



    # print(customers_data.columns.get_level_values("id").unique().size)
    # print(customers_data.columns.get_level_values("attribute").unique())

    customers_embedding, targets_embedding, neighbors_embedding, neighbors_mask, neighbors_type_embedding, customer_id_to_idx, \
    customers_with_out_degree, customers_without_out_degree = extract_dateframe(customers_data, G)

    print("num training customers: {}".format(len(customers_with_out_degree)))
    print("num not training customers: {}".format(len(customers_without_out_degree)))

    torch.save(customers_embedding, path.join(BASE_DIR, "customers_embed.pt"))
    torch.save(targets_embedding, path.join(BASE_DIR, "targets_embed.pt"))
    torch.save(neighbors_embedding, path.join(BASE_DIR, "neighbors_embed.pt"))
    torch.save(neighbors_mask, path.join(BASE_DIR, "neighbors_msk.pt"))
    torch.save(neighbors_type_embedding, path.join(BASE_DIR, "neighbors_type.pt"))
    pickle.dump(customer_id_to_idx, open(path.join(BASE_DIR, "customers_id_to_idx.bin"), "wb"))
    pickle.dump(customers_with_out_degree, open(path.join(BASE_DIR, "customers_with_out_degree.bin"), "wb"))
    pickle.dump(customers_without_out_degree, open(path.join(BASE_DIR, "customers_without_out_degree.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(customers_with_out_degree, customer_id_to_idx, 2000)

    pickle.dump(train_dataset, open(path.join(BASE_DIR, "train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "test_dataset.bin"), "wb"))




