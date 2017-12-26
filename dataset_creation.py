import mysql.connector
import pickle
from os.path import join as path_join
from collections import OrderedDict
from bidict import bidict
import numpy as np
from datetime import datetime
import random
from helper import TIMESTAMP, T_risk, T_attribute, RISK_ATTRIBUTE, C_ATTRIBUTE, REF_DATE, DATE_FORMAT

config = {
  'user': 'root',
  'password': 'vela1990',
  'host': '127.0.0.1',
  'database': 'ml_crif',
}

GET_ALL_CUSTOMER_ID = "SELECT DISTINCT customerid FROM customers"
GET_ALL_OWNER_ID = "SELECT DISTINCT customerid FROM onemancompany_owners"
GET_ALL_RISK_ID = "SELECT DISTINCT customerid FROM risk"
GET_ALL_REVENUE_ID = "SELECT DISTINCT customerid FROM revenue"
GET_ALL_CUSTOMER_LINKS_ID = "SELECT DISTINCT * FROM (SELECT c_one.customerid FROM customer_links AS c_one UNION SELECT c2.customerid_link FROM customer_links AS c2) AS u"
GET_ALL_RISK_ID_ONEMANCOMPANY = "SELECT DISTINCT r.customerid FROM risk AS r, onemancompany_owners AS oc WHERE r.customerid = oc.customerid"
GET_ALL_REVENUE_ID_ONEMANCOMPANY = "SELECT DISTINCT r.customerid FROM revenue AS r, onemancompany_owners AS oc WHERE r.customerid = oc.customerid"
CUSTOMERS_OWNER_UNION_ID = "SELECT c.customerid FROM customers AS c UNION SELECT o.customerid FROM onemancompany_owners AS o"

GET_ALL_RISK_USER_AND_LINKS = 'SELECT cl.customerid, cl.customerid_link, cl.cod_link_type, cl.des_link_type, r.date_ref, r.val_scoring_risk, r.class_scoring_risk, r.val_scoring_ai, r.class_scoring_ai, r.val_scoring_cr, r.class_scoring_cr, r.val_scoring_bi, r.class_scoring_bi, r.val_scoring_sd, r.class_scoring_sd, r.pre_notching FROM risk AS r, customer_links AS cl WHERE r.customerid = cl.customerid'
GET_ALL_RISK_LINKS = "SELECT DISTINCT cl.customerid, cl.customerid_link, cl.cod_link_type,  cl.des_link_type FROM risk AS r, customer_links AS cl WHERE r.customerid = cl.customerid"

GET_ALL_CUSTOMER_LINKS = "SELECT * FROM customer_links"
GET_ALL_RISK = "SELECT customerid, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching FROM risk"
GET_ALL_CUSTOMER = "SELECT customerid, birthdate, b_partner, cod_uo, zipcode, region, country_code, c.customer_kind, ck.description as kind_desc, c.customer_type, ct.description as type_desc, uncollectible_status, ateco, sae  FROM customers as c, customer_kinds as ck, customer_types as ct WHERE c.customer_kind=ck.customer_kind AND c.customer_type = ct.customer_type"
GET_RISK_BY_CUSTOMER_ID = "SELECT customerid, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching FROM risk WHERE customerid = {}"
GET_REVENUE_BY_CUSTOMER_ID = "SELECT customerid, date_ref, val_scoring_rev, class_scoring_rev,  val_scoring_op, class_scoring_op,  val_scoring_co, class_scoring_co    FROM revenue WHERE customerid = {}"
GET_ALL_ONEMANCOMPANY = "SELECT customerid, customerid_join FROM onemancompany_owners"
GET_ATECO = "SELECT * FROM ateco"
GET_SAE = "SELECT * FROM sae"


ATECO_DICT = bidict({})
B_PARTNER_DICT = bidict({})
COD_UO_DICT = bidict({})
COUNTRY_CODE_DICT = bidict({})
C_KIND_DICT = bidict({})
C_TYPE_DICT = bidict({})
REGION_DICT = bidict({})
SAE_DICT = bidict({})
US_DICT = bidict({})
ZIPCODE_DICT = bidict({})
SEGMENTO_DICT = bidict({})

def get_ateco_description():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    ret = {}
    try:
        cursor.execute(GET_ATECO)
        for cod_ateco, description in cursor:
            ret[cod_ateco] = description
    finally:
        cnx.close()
    return ret

def get_sea_description():
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    ret = {}
    try:
        cursor.execute(GET_SAE)
        for cod_sae, description in cursor:
            ret[cod_sae] = description
    finally:
        cnx.close()
    return ret


def get_value(id, dict):
    """
    Compute the value for the given input id
    :param id:
    :param dict:
    :return:
    """
    if id in dict:
        value = dict[id]
    else:
        value = len(dict)
        dict[id] = value
    return value

def get_day_diff(birth_date):
    '''
    compute day difference between REF_DATE and the birth day
    :param birth_date:
    :return:
    '''
    r_date = datetime.strptime(REF_DATE, DATE_FORMAT)
    b_date = datetime.strptime(birth_date, DATE_FORMAT)
    return (r_date - b_date).days / 7

def fix_ateco_code(ateco):
    return ateco.replace("X", "")


def format_risk(risk):
    check_null = lambda x: -10 if np.isnan(x) else x

    formatted_risk = OrderedDict()


    # extract timestemp
    for timestemp in TIMESTAMP:
        # try:
        t_risk = risk[timestemp]

        # check attribute for give timestemp
        t_risk_value = []
        for r_attribute in RISK_ATTRIBUTE:
            if r_attribute == "segmento":
                if t_risk[r_attribute] == "UNR" or t_risk[r_attribute] == "UNT":
                    raise KeyError("segmento-{}".format(t_risk[r_attribute]))
                else:
                    t_risk_value.append(get_value(t_risk[r_attribute], SEGMENTO_DICT))
            else:
                t_risk_value.append(check_null(t_risk[r_attribute]))

        formatted_t_risk = T_risk._make(t_risk_value)

        # save formatted timestemp
        formatted_risk[timestemp] = formatted_t_risk
        # except KeyError as ke:
        #     print("id:{}\tKey: {}".format(customer_id, ke))

    return formatted_risk

def format_attribute(attribute, ateco_des, sea_des):
    ateco, ateco_des = attribute["ateco"], ateco_des[attribute["ateco"]]
    b_partner = attribute['b_partner']
    birth_date = attribute['birth_date']
    cod_uo = attribute['cod_uo']
    country_code = attribute['country_code']
    c_kind, k_desc = attribute['customer_kind'], attribute['kind_desc']
    c_type, t_desc = attribute['customer_type'], attribute['type_desc']
    region = attribute['region']
    sae, sae_des = attribute['sae'], sea_des[attribute['sae']]
    uncollectable_status = attribute['uncollectable_status']
    zipcode = attribute['zipcode']

    atribute_values = [
        get_value((ateco, ateco_des), ATECO_DICT),
        get_value(b_partner, B_PARTNER_DICT),
        get_day_diff(birth_date),
        get_value(cod_uo, COD_UO_DICT),
        get_value(country_code, COUNTRY_CODE_DICT),
        get_value((c_kind, k_desc), C_KIND_DICT),
        get_value((c_type, t_desc), C_TYPE_DICT),
        get_value(region, REGION_DICT),
        get_value((sae, sae_des), SAE_DICT),
        get_value(uncollectable_status, US_DICT),
        get_value(zipcode, ZIPCODE_DICT),
    ]
    formatted_attribute = T_attribute._make(atribute_values)
    return formatted_attribute


if __name__ == "__main__":
    customer_data = pickle.load(open(path_join("./data", "customers", "customers_attribute_risk_neighbor.bin"), "rb"))
    customer_formated_data = {}

    ateco_des = get_ateco_description()
    sea_des = get_sea_description()
    for row, customer_id in enumerate(sorted(customer_data.keys())):
        try:
            customer_risk, customer_attribute = customer_data[customer_id]["risk_attribute"], customer_data[customer_id]["node_attribute"]
            customer_attribute["ateco"] = fix_ateco_code(customer_attribute["ateco"])

            customer_risk = format_risk(customer_risk)
            customer_attribute = format_attribute(customer_attribute, ateco_des, sea_des)

            customer_formated_data[customer_id] = dict(risk=customer_risk,
                                                       attribute=customer_attribute)
        except KeyError as ke:
            print("{}\t{}".format(customer_id, ke))
        except Exception as e:
            raise e

        if row % 100 == 0:
            print(row)

    print(len(customer_formated_data))

    pickle.dump(ATECO_DICT, open(path_join("./data", "customers", "dicts", "ateco_dict.bin"), "wb"))
    pickle.dump(B_PARTNER_DICT, open(path_join("./data", "customers", "dicts", "b_partner_dict.bin"), "wb"))
    pickle.dump(COD_UO_DICT, open(path_join("./data", "customers", "dicts", "cod_uo_dict.bin"), "wb"))
    pickle.dump(COUNTRY_CODE_DICT, open(path_join("./data", "customers", "dicts", "country_code_dict.bin"), "wb"))
    pickle.dump(C_KIND_DICT, open(path_join("./data", "customers", "dicts", "c_kind_dict.bin"), "wb"))
    pickle.dump(C_TYPE_DICT, open(path_join("./data", "customers", "dicts", "c_type_dict.bin"), "wb"))
    pickle.dump(REGION_DICT, open(path_join("./data", "customers", "dicts", "region_dict.bin"), "wb"))
    pickle.dump(SAE_DICT, open(path_join("./data", "customers", "dicts", "sae_dict.bin"), "wb"))
    pickle.dump(US_DICT, open(path_join("./data", "customers", "dicts", "uncollectable_status_dict.bin"), "wb"))
    pickle.dump(ZIPCODE_DICT, open(path_join("./data", "customers", "dicts", "zipcode_dict.bin"), "wb"))
    pickle.dump(SEGMENTO_DICT, open(path_join("./data", "customers", "dicts", "segmento_dict.bin"), "wb"))

    pickle.dump(customer_formated_data, open(path_join("./data", "customers", "customers_formatted_attribute_risk.bin"), "wb"))

    # customer_formated_data = pickle.load(open(path_join("./data", "customers", "customers_formatted_attribute_risk.bin"), "rb"))

    training_sample = random.sample(customer_formated_data.items(), 18000)
    training_data = []
    for c_id, att_dict in training_sample:
        training_data.append(dict(customer_id=c_id, risk=att_dict['risk'], attribute=att_dict['attribute']))
        del customer_formated_data[c_id]

    eval_sample = random.sample(customer_formated_data.items(), 4000)
    eval_data = []
    for c_id, att_dict in eval_sample:
        eval_data.append(dict(customer_id=c_id, risk=att_dict['risk'], attribute=att_dict['attribute']))
        del customer_formated_data[c_id]

    test_data = []
    for c_id, att_dict in customer_formated_data.items():
        test_data.append(dict(customer_id=c_id, risk=att_dict['risk'], attribute=att_dict['attribute']))



    pickle.dump(training_data,
                open(path_join("./data", "customers", "train_customers_formatted_attribute_risk.bin"), "wb"))

    pickle.dump(eval_data,
                open(path_join("./data", "customers", "eval_customers_formatted_attribute_risk.bin"), "wb"))

    pickle.dump(test_data,
                open(path_join("./data", "customers", "test_customers_formatted_attribute_risk.bin"), "wb"))





