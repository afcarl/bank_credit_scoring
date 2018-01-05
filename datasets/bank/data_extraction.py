import mysql.connector
import pickle
from collections import OrderedDict
import numpy as np
import visdom
from itertools import tee
from bidict import bidict
from datetime import datetime
from os.path import join as path_join
vis = visdom.Visdom()

config = {
    'user': 'root',
    'password': 'vela1990',
    'host': '127.0.0.1',
    'database': 'ml_crif',
}

TIMESTAMP = ["2016-06-30", "2016-07-31", "2016-08-31", "2016-09-30", "2016-10-31", "2016-11-30", "2016-12-31",
             "2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30", "2017-05-31", "2017-06-30"]

REF_DATE = "20180101"
DATE_FORMAT = "%Y%m%d"

GET_ALL_CUSTOMER = "SELECT customerid FROM customers"
GET_ALL_OWNER = "SELECT customerid FROM onemancompany_owners"
CUSTOMERS_OWNER_UNION = "SELECT c.customerid FROM customers AS c UNION SELECT o.customerid FROM onemancompany_owners AS o"
GET_REVENUE_USER = "SELECT customerid FROM revenue"
GET_RISK_USER = "SELECT customerid, segmento, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching  FROM risk ORDER BY customerid asc, date_ref asc"
GET_RISK_USER_BY_ID = "SELECT customerid, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching  FROM risk ORDER BY date_ref asc WHERE customerid={}"
GET_ALL_CUSTOMER_LINKS_ID = "SELECT DISTINCT * FROM (SELECT c_one.customerid FROM customer_links AS c_one UNION SELECT c2.customerid_link FROM customer_links AS c2) AS u"
GET_ALL_CUSTOMER_LINKS_BY_ID = "SELECT DISTINCT customerid_link FROM customer_links WHERE customerid={}"
GET_ALL_RISK_LINKS_BY_CUSTOMERID = "SELECT DISTINCT cl.customerid, cl.customerid_link, cl.cod_link_type,  cl.des_link_type FROM risk AS r, customer_links AS cl WHERE r.customerid = cl.customerid AND r.customerid={}"
GET_DEFAULT_RISK_CUSTOMER = "SELECT r.customerid, r.date_ref, r.val_scoring_risk, r.class_scoring_risk, r.val_scoring_pre, r.class_scoring_pre, r.val_scoring_ai, r.class_scoring_ai, r.val_scoring_cr, r.class_scoring_cr, r.val_scoring_bi, r.class_scoring_bi, r.val_scoring_sd, r.class_scoring_sd, r.pre_notching  FROM risk AS r  WHERE r.customerid IN (SELECT DISTINCT r1.customerid FROM ml_crif.risk AS r1 WHERE r1.val_scoring_risk=100) ORDER BY r.customerid asc, r.date_ref asc"
GET_CUSTOMER_BY_ID = "SELECT birthdate, b_partner, cod_uo, zipcode, region, country_code, c.customer_kind, ck.description as kind_desc, c.customer_type, ct.description as type_desc, uncollectible_status, ateco, sae  FROM customers as c, customer_kinds as ck, customer_types as ct WHERE c.customer_kind=ck.customer_kind AND c.customer_type = ct.customer_type AND c.customerid={}"

f_check_none = lambda x: np.nan if x == None else x
f_parse_date = lambda x: "{}-{}-{}".format(x[:4], x[4:6], x[6:])
f_check_b_date = lambda x: REF_DATE if x == "" else x

def extract_default_customers_timeseries(cursor):
    # cursor.execute(GET_RISK_USER)
    # customers = {}
    # for row, (customer_id, segmento, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre,
    #           val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi,
    #           val_scoring_sd, class_scoring_sd, pre_notching) in enumerate(cursor):
    #
    #     if customer_id in customers:
    #         risk_attribute = customers[customer_id]["risk_attribute"]
    #         risk_mapping(risk_attribute, segmento, f_parse_date(date_ref), val_scoring_risk, class_scoring_risk,
    #                      val_scoring_pre, class_scoring_pre,
    #                      val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi,
    #                      class_scoring_bi,
    #                      val_scoring_sd, class_scoring_sd, pre_notching)
    #     else:
    #         risk_attribute = OrderedDict()
    #         risk_mapping(risk_attribute, segmento, f_parse_date(date_ref), val_scoring_risk, class_scoring_risk,
    #                      val_scoring_pre, class_scoring_pre,
    #                      val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi,
    #                      class_scoring_bi,
    #                      val_scoring_sd, class_scoring_sd, pre_notching)
    #         customers[customer_id] = dict(risk_attribute=risk_attribute)
    #
    #     if row % 100 == 0:
    #         print(row)
    # print(len(customers))
    # pickle.dump(customers, open(path_join("data", "customers", "customers_risk.bin"), "wb"))
    # for row, customer_id in enumerate(sorted(customers.keys())):
    #     cursor.execute(GET_CUSTOMER_BY_ID.format(customer_id))
    #     done = False
    #     for birth_date, b_partner, cod_uo, zipcode, region, country_code, customer_kind, kind_desc, customer_type, type_desc, uncollectable_status, ateco, sae in cursor:
    #         node_attribute = dict(
    #             birth_date=f_parse_date(f_check_b_date(birth_date)),
    #             b_partner=b_partner,
    #             cod_uo=cod_uo,
    #             zipcode=zipcode,
    #             region=region,
    #             country_code=country_code,
    #             customer_kind=customer_kind,
    #             kind_desc=kind_desc,
    #             customer_type=customer_type,
    #             type_desc=type_desc,
    #             uncollectable_status=uncollectable_status,
    #             ateco=ateco,
    #             sae=sae
    #         )
    #         customers[customer_id]["node_attribute"] = node_attribute
    #         done = True
    #     if not done:
    #         del customers[customer_id]
    #     if row % 100 == 0:
    #         print(row)
    # print(len(customers))
    # pickle.dump(customers, open(path_join("data", "customers", "customers_attribute_risk.bin"), "wb"))
    customers = pickle.load(open(path_join("data", "customers", "customers_attribute_risk.bin"), "rb"))
    for row, (customer_id, customers_attributes) in enumerate(customers.items()):
        cursor.execute(GET_ALL_CUSTOMER_LINKS_BY_ID.format(customer_id))
        neighbors = []
        for neighbor, in cursor:
            if neighbor in customers:
                neighbors.append(neighbor)
        customers_attributes["neighbor"] = neighbors
        if row % 100 == 0:
            print(row)

    pickle.dump(customers, open(path_join("data", "customers", "customers_attribute_risk_neighbor.bin"), "wb"))




def risk_mapping(risk_attribute, segmento, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre,
                 val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi,
                 val_scoring_sd, class_scoring_sd, pre_notching):


    risk_attribute[date_ref] = {
        "segmento": segmento,
        "val_scoring_risk": f_check_none(val_scoring_risk),
        "class_scoring_risk": class_scoring_risk,
        "val_scoring_pre": f_check_none(val_scoring_pre),
        "class_scoring_pre": class_scoring_pre,
        "val_scoring_ai": f_check_none(val_scoring_ai),
        "class_scoring_ai": class_scoring_ai,
        "val_scoring_cr": f_check_none(val_scoring_cr),
        "class_scoring_cr": class_scoring_cr,
        "val_scoring_bi": f_check_none(val_scoring_bi),
        "class_scoring_bi": class_scoring_bi,
        "val_scoring_sd": f_check_none(val_scoring_sd),
        "class_scoring_sd": class_scoring_sd,
        "pre_notching": pre_notching
    }

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
        extract_default_customers_timeseries(cursor)

    finally:
        cursor.close()
        cnx.close()