import mysql.connector
import pickle
from collections import OrderedDict
import numpy as np
import visdom
import pandas as pd
from itertools import tee
from bidict import bidict
from datetime import datetime
import os.path as path
from itertools import islice
import msgpack


BASE_DIR = path.join("..", "..", "data", "customers")

vis = visdom.Visdom()

config = {
    'user': 'root',
    'password': 'vela1990',
    'host': '127.0.0.1',
    'database': 'ml_crif',
}

OBJ_COLUMNS = ['segmento', 'date_ref', 'b_partner', 'cod_uo', 'zipcode', 'region', 'country_code', 'customer_kind', 'kind_desc', 'customer_type', 'type_desc', 'uncollectable_status', 'ateco', 'sae']
RISK_COLUMNS = ["segmento", "date_ref", "pre_notching", "val_scoring_risk", "val_scoring_pre", "val_scoring_ai", "val_scoring_cr", "val_scoring_bi", "val_scoring_sd", "class_scoring_risk", "class_scoring_pre", "class_scoring_ai", "class_scoring_cr", "class_scoring_bi", "class_scoring_sd"]
TIMESTAMP = ["2016-06-30", "2016-07-31", "2016-08-31", "2016-09-30", "2016-10-31", "2016-11-30", "2016-12-31",
             "2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30", "2017-05-31", "2017-06-30"]


ONE_MAN_COMPANY_COSTUMERS = [13379, 49098, 66357, 66410, 228463, 392729, 394761, 418783, 424879, 430555, 434466, 1356061, 4316517, 4320054, 5280859, 5383259, 5384885, 5390867]
REF_DATE = "20170101"
DATE_FORMAT = "%Y%m%d"

GET_ALL_CUSTOMER = "SELECT customerid FROM customers"
GET_ALL_OWNER = "SELECT customerid FROM onemancompany_owners"
CUSTOMERS_OWNER_UNION = "SELECT c.customerid FROM customers AS c UNION SELECT o.customerid FROM onemancompany_owners AS o"
GET_REVENUE_USER = "SELECT customerid FROM revenue"
GET_RISK_USER = "SELECT customerid, segmento, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching  FROM risk ORDER BY customerid asc, date_ref asc"
GET_RISK_USER_BY_ID = "SELECT customerid, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching  FROM risk ORDER BY date_ref asc WHERE customerid={}"
GET_ALL_CUSTOMER_LINKS_ID = "SELECT DISTINCT * FROM (SELECT c_one.customerid FROM customer_links AS c_one UNION SELECT c2.customerid_link FROM customer_links AS c2) AS u"
GET_ALL_CUSTOMER_LINKS_BY_ID = "SELECT DISTINCT customerid_link FROM customer_links WHERE customerid={}"
GET_ALL_CUSTOMER_LINKS_FOR_RISK_CUSTOMERS = "SELECT DISTINCT cl.customerid, customerid_link FROM customer_links as cl, risk as r WHERE r.customerid = cl.customerid"
GET_PAGE_CUSTOMER_LINKS_FOR_RISK_CUSTOMERS = "SELECT DISTINCT cl.customerid, customerid_link FROM customer_links as cl, risk as r WHERE r.customerid = cl.customerid and r.customerid > {} LIMIT 1000"
GET_ALL_RISK_LINKS_BY_CUSTOMERID = "SELECT DISTINCT cl.customerid, cl.customerid_link, cl.cod_link_type,  cl.des_link_type FROM risk AS r, customer_links AS cl WHERE r.customerid = cl.customerid AND r.customerid={}"
GET_DEFAULT_RISK_CUSTOMER = "SELECT r.customerid, r.date_ref, r.val_scoring_risk, r.class_scoring_risk, r.val_scoring_pre, r.class_scoring_pre, r.val_scoring_ai, r.class_scoring_ai, r.val_scoring_cr, r.class_scoring_cr, r.val_scoring_bi, r.class_scoring_bi, r.val_scoring_sd, r.class_scoring_sd, r.pre_notching  FROM risk AS r  WHERE r.customerid IN (SELECT DISTINCT r1.customerid FROM ml_crif.risk AS r1 WHERE r1.val_scoring_risk=100) ORDER BY r.customerid asc, r.date_ref asc"
GET_CUSTOMER_BY_ID = "SELECT birthdate, b_partner, cod_uo, zipcode, region, country_code, c.customer_kind, ck.description as kind_desc, c.customer_type, ct.description as type_desc, uncollectible_status, ateco, sae  FROM customers as c, customer_kinds as ck, customer_types as ct WHERE c.customer_kind=ck.customer_kind AND c.customer_type = ct.customer_type AND c.customerid={} LIMIT 0, 1"

f_check_none = lambda x: np.nan if x == None else x
f_parse_date = lambda x: "{}-{}-{}".format(x[6:], x[4:6], x[:4])
f_format_str_date = lambda x: datetime.strptime(x, "%d-%m-%Y")
f_check_b_date = lambda x: REF_DATE if x == "" else x


def chunks(data, SIZE=30):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def take(data_dict, N_first=30):
    it = iter(data_dict)
    return {k:data_dict[k] for k in islice(it, N_first)}

def calculate_age(born):
    today = datetime.now()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def remove_customers_with_no_neighbors(customers_data, customers_neighbors):
    n_customers_id = set(customers_neighbors.keys())
    df_customers_id = customers_data.columns.get_level_values("id").unique()
    print("df_customer len: {}".format(len(df_customers_id)))
    print("n_customer len: {}".format(len(n_customers_id)))
    for row, customer_id in enumerate(df_customers_id):
        if customer_id not in n_customers_id:
            customers_data = customers_data.drop(customer_id, axis=1, level='id')
        if (row % 100) == 0:
            print(row, len(customers_data.columns.get_level_values("id").unique()))
    print("final df_customers: {}".format(len(customers_data.columns.get_level_values("id").unique())))
    return customers_data

def get_customers_risk(cursor):
    """
    get the customers whit risk values
    :param cursor:
    :return:
    """
    customers_data = OrderedDict()
    prev_customer_id = 0
    for row, (customer_id, segmento, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre,
              val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi,
              val_scoring_sd, class_scoring_sd, pre_notching) in enumerate(cursor):

        if not customer_id in customers_data and customer_id != prev_customer_id:
            if row > 0:
                customers_data[prev_customer_id] = pd.DataFrame.from_dict(customers_data[prev_customer_id], orient="index")
            customers_data[customer_id] = OrderedDict()
            prev_customer_id = customer_id

        date_ref = f_parse_date(date_ref)
        customers_data[customer_id][f_format_str_date(date_ref)] = __risk_mapping__(segmento, date_ref, pre_notching,
                                                                                    val_scoring_risk, val_scoring_pre, val_scoring_ai, val_scoring_cr, val_scoring_bi, val_scoring_sd,
                                                                                    class_scoring_risk, class_scoring_pre, class_scoring_ai, class_scoring_cr, class_scoring_bi, class_scoring_sd)

        if row % 100 == 0:
            print(row)
    customers_data[prev_customer_id] = pd.DataFrame.from_dict(customers_data[prev_customer_id], orient="index")
    print(len(customers_data))
    pickle.dump(customers_data, open(path.join(BASE_DIR, "temp", "customers_risk_dict.bin"), "wb"))
    return customers_data

def get_neighbors(customers_data, cursor):
    """
    get the neighbors that has some risk values of nodes with risk value
    :param customers_data:
    :param cursor:
    :return:
    """
    customers_neighbors = OrderedDict()
    for row, customer_id in enumerate(customers_data.keys()):
        cursor.execute(GET_ALL_CUSTOMER_LINKS_BY_ID.format(customer_id))
        for customer_id_link, in cursor:
            if customer_id_link in customers_data:
                if customer_id in customers_neighbors:
                    customers_neighbors[customer_id].append(customer_id_link)
                else:
                    customers_neighbors[customer_id] = [customer_id_link]

        if row % 100 == 0:
            print("row:{}\tcustomers:{}".format(row, len(customers_neighbors)))

    print("row:{}\tcustomers:{}".format(row, len(customers_neighbors)))
    pickle.dump(customers_neighbors, open(path.join(BASE_DIR, "temp", "customers_neighbors_dict.bin"), "wb"))
    return customers_neighbors

def __get_customer_info__(customer_id, cursor):
    cursor.execute(GET_CUSTOMER_BY_ID.format(customer_id))
    birth_date, b_partner, cod_uo, zipcode, region, country_code, customer_kind, kind_desc, customer_type, type_desc, uncollectable_status, ateco, sae = cursor.fetchone()
    attribute = dict(
        age=calculate_age(f_format_str_date(f_parse_date(f_check_b_date(birth_date)))),
        b_partner=b_partner,
        cod_uo=cod_uo,
        zipcode=zipcode,
        region=region,
        country_code=country_code,
        customer_kind=customer_kind,
        kind_desc=kind_desc,
        customer_type=customer_type,
        type_desc=type_desc,
        uncollectable_status=uncollectable_status,
        ateco=ateco,
        sae=sae)
    return attribute

def get_customers_info(customers_data, cursor):
    """
    get the customers general attribute
    :param customers_data:
    :param cursor:
    :return:
    """
    num_customer = len(customers_data)
    for row, (customer_id, df) in enumerate(sorted(customers_data.items())):
        node_attribute = {}
        attribute = __get_customer_info__(customer_id, cursor)
        for time_step in df.index:
            node_attribute[time_step] = attribute
        node_attribute = pd.DataFrame.from_dict(node_attribute, orient="index")
        assert not node_attribute.isnull().values.any(), "{}\n{}".format(customer_id, node_attribute)
        customers_data[customer_id] = pd.concat([df, node_attribute], axis=1)
        print(row, customer_id)
    assert len(customers_data) == num_customer, "old lenght:{}\t newlenght:{}".format(num_customer, len(customers_data))

    pickle.dump(customers_data, open(path.join(BASE_DIR, "temp", "customers_risk_attribute_dict.bin"), "wb"))
    return customers_data

def __risk_mapping__(segmento, date_ref, pre_notching,
                 val_scoring_risk, val_scoring_pre, val_scoring_ai, val_scoring_cr, val_scoring_bi, val_scoring_sd,
                 class_scoring_risk, class_scoring_pre, class_scoring_ai, class_scoring_cr, class_scoring_bi, class_scoring_sd):
    return {
        "segmento": segmento,
        "date_ref": date_ref,
        "pre_notching": pre_notching,
        "val_scoring_risk": f_check_none(val_scoring_risk),
        "val_scoring_pre": f_check_none(val_scoring_pre),
        "val_scoring_ai": f_check_none(val_scoring_ai),
        "val_scoring_cr": f_check_none(val_scoring_cr),
        "val_scoring_bi": f_check_none(val_scoring_bi),
        "val_scoring_sd": f_check_none(val_scoring_sd),
        "class_scoring_risk": class_scoring_risk,
        "class_scoring_pre": class_scoring_pre,
        "class_scoring_ai": class_scoring_ai,
        "class_scoring_cr": class_scoring_cr,
        "class_scoring_bi": class_scoring_bi,
        "class_scoring_sd": class_scoring_sd
    }


def cut_time_series(customers_data, max_num_nan=4, nan_replacement=-1):
    customers_data = customers_data.iloc[2:]   # remove 2015 data

    full_nan_row = customers_data.isnull().all(axis=1, level="id")      # get full nan row for each id
    delete_ids = full_nan_row.sum().loc[lambda x: x > max_num_nan].keys()   # get customers to delete
    customers_data = customers_data.drop(delete_ids, axis=1)            # delete obtained columns

    # replace nan row with previous values
    full_nan_row = customers_data.isnull().all(axis=1, level="id")
    customers_to_fill = full_nan_row.any().loc[lambda x: x == True].index
    customers_data.update(customers_data[customers_to_fill].fillna(method='bfill').fillna(method='ffill'))

    # replace nan in RISK_COLUMN by -1
    customers_data.update(customers_data.loc[:, pd.IndexSlice[:, RISK_COLUMNS]].fillna(nan_replacement))
    nan_customers = customers_data.isnull().any(axis=1, level="id").any()
    return customers_data, nan_customers


def check_dataframe_dim(customers_data, expected_size=(18,28)):
    """
    check why some customers do not have the same dimension.
    One-man-company do not have customers attribute.
    FIX it by inserting the co-owner information extracted form the customers link table
    :param customers_data: datframe to check
    :param expected_size: expected dim
    :return:
    """
    print("shape: {}".format(customers_data.shape))
    print("customers: {}".format(customers_data.columns.get_level_values('id').unique().shape[0]))
    print("attribute number: {}".format(customers_data.columns.get_level_values('attribute').unique().shape[0]))

    ret = []
    for customer_id in customers_data.columns.get_level_values('id').unique():
        if customers_data[customer_id].shape != expected_size:
            print("{}\terror".format(customer_id))
            print(customers_data[customer_id])
            ret.append(customer_id)
    print("len:{}\n{}".format(len(ret), ret))
    return ret

def delete_one_man_company(customers_data, customers_neighbors_dict, id_to_delete=ONE_MAN_COMPANY_COSTUMERS):
    """
    delete company with no customers information
    :param customers_data:
    :param customers_neighbors_dict:
    :param id_to_delete:
    :return:
    """
    deleted_costumers = []
    for customer_id in id_to_delete:
        if customer_id in deleted_costumers:
            print("already deleted")
            continue
        customers_data = customers_data.drop(customer_id, axis=1, level=0)
        neighbors = customers_neighbors_dict.pop(customer_id)
        deleted_costumers.append(customer_id)
        for neighbor in neighbors:
            if neighbor in customers_neighbors_dict:
                customers_neighbors_dict[neighbor].remove(customer_id)
                if len(customers_neighbors_dict[neighbor]) == 0:
                    customers_data = customers_data.drop(neighbor, axis=1, level=0)
                    customers_neighbors_dict.pop(neighbor)
                    deleted_costumers.append(neighbor)

    return customers_data, customers_neighbors_dict

def fix_neighbors(customers_data, customers_neighbors):
    """
    remove from the neighbors set the customers not presenet in the data
    :param customers_data:
    :param customers_neighbors:
    :return:
    """
    def __check_customer__(to_check, customers_set):
        for customer_id in to_check:
            if customer_id not in customers_set:
                to_check.remove(customer_id)
        return to_check

    removed = []
    customers = customers_data.columns.get_level_values('id').unique().tolist()

    for customer_id in list(customers_neighbors.keys()):
        if customer_id in customers:
            neighbors = __check_customer__(customers_neighbors[customer_id], customers)
            if len(neighbors) == 0:
                print("{} without neighbors".format(customer_id))
                customers.remove(customer_id)
                customers_neighbors.pop(customer_id)
                removed.append(customer_id)
            else:
                customers_neighbors[customer_id] = neighbors
        else:
            customers_neighbors.pop(customer_id)

    assert len(customers) == len(customers_neighbors), "{}\t{}".format(len(customers), len(customers_neighbors))
    assert len(customers) + len(removed) == customers_data.columns.get_level_values('id').unique().shape[0], "{}\t{}\{}".format(len(customers), len(removed), customers_data.columns.get_level_values('id').unique().shape[0])

    for customer_id in removed:
        customers_data = customers_data.drop(customer_id, axis=1, level=0)
    return customers_data, customers_neighbors



def extract_data(cursor):

    # customers_data = get_customers_risk(cursor)
    # customers_neighbors = get_neighbors(customers_data, cursor)
    # customers_data = get_customers_info(customers_data, cursor)
    # customers_data = pd.concat(customers_data, axis=1)
    # customers_data.columns = customers_data.columns.rename(['id', 'attribute'])
    # customers_data = remove_customers_with_no_neighbors(customers_data, customers_neighbors)
    # customers_data.to_msgpack(path.join(BASE_DIR, "temp", "customers_risk_df.msg"))

    # customers_data, nan_customers = cut_time_series(customers_data)
    # customers_data.to_msgpack(path.join(BASE_DIR, "temp", "customers_risk_time_frame_null_df.msg"))

    # check_dataframe_dim(customers_data)
    # customers_data = delete_one_man_company(customers_data, customers_neighbors_dict)
    # customers_data, customers_neighbors_dict = fix_neighbors(customers_data, customers_neighbors_dict)


    customers_data = pd.read_msgpack(path.join(BASE_DIR, "temp", "customers_risk_time_frame_null_df_final.msg"))
    with open(path.join(BASE_DIR, "temp", "customers_neighbors_dict.msg"), 'rb') as infile:
        customers_neighbors_dict = msgpack.unpack(infile)

    print(customers_data.columns.get_level_values('id').unique().shape[0])
    print(len(customers_neighbors_dict))


def extract_neighborhod_risk():
    customer_data = pickle.load(open("customer_risk_time.bin", "rb"))
    customer_origin_data = OrderedDict()
    customer_diff_data = OrderedDict()
    customer_rel_diff_data = OrderedDict()

    n_done = 0
    tot = len(customer_data.keys())
    for id, cusomter_id in enumerate(sorted(customer_data.keys())):
        customer_risk = customer_data[cusomter_id]
        c_risk = np.array(customer_risk)
        c_r_risk = np.diff(c_risk, axis=0)

        cursor.execute(GET_ALL_CUSTOMER_LINKS_BY_ID.format(cusomter_id))
        for customer_link_id, in cursor:
            if customer_link_id in customer_data:
                neiborhod_risk = customer_data[customer_link_id]
                n_risk = np.array(neiborhod_risk)
                n_r_risk = np.diff(n_risk, axis=0)

                # save original data
                if cusomter_id in customer_origin_data:
                    customer_origin_data[cusomter_id].append(n_risk)
                else:
                    customer_origin_data[cusomter_id] = [n_risk]

                # compute absolute difference
                diff = np.fabs(np.array(c_risk) - np.array(n_risk))
                if cusomter_id in customer_diff_data:
                    customer_diff_data[cusomter_id].append(diff)
                else:
                    customer_diff_data[cusomter_id] = [diff]

                # compute relative difference
                diff = np.fabs(np.array(c_r_risk) - np.array(n_r_risk))
                if cusomter_id in customer_rel_diff_data:
                    customer_rel_diff_data[cusomter_id].append(diff)
                else:
                    customer_rel_diff_data[cusomter_id] = [diff]
        n_done += 1
        if id % 100 == 0:
            done = (n_done / tot) * 100
            print(done)
    return customer_origin_data, customer_diff_data, customer_rel_diff_data


if __name__ == "__main__":
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    try:
        extract_data(cursor)

    finally:
        cursor.close()
        cnx.close()