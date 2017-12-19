import mysql.connector
from mysql.connector import errorcode
import networkx as nx
import pickle
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


cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()



if __name__ == "__main__":
    G = nx.DiGraph()
    try:

        f_check_none = lambda x: -1 if x == None else x

        # create graph
        # cursor.execute(GET_ALL_RISK_LINKS)
        # for customer_id, customer_link, cond_link_type, des_link_type in cursor:
        #     G.add_edge(customer_id, customer_link, cod_type=cond_link_type, des_link_type=des_link_type)
        #
        # print("total nodes: {}".format(G.number_of_nodes()))
        # print("total edges: {}".format(G.number_of_edges()))
        # print("average degree: {}".format(nx.average_degree_connectivity(G)))
        # pickle.dump(G, open("graph_only_risk.bin", "wb"))
        #
        # # add risk attribute
        # cursor.execute(GET_ALL_RISK_ID)
        # risk_ids = []
        # counter_risk_id = 0
        # for id, in cursor:
        #     risk_ids.append(id)
        # # add risk attribute
        # for id in risk_ids:
        #     if G.has_node(id):
        #         if 'risk_attribute' in G.nodes[id]:
        #             counter_risk_id += 1
        #         else:
        #             cursor.execute(GET_RISK_BY_CUSTOMER_ID.format(id))
        #             time_stamps = []
        #             for customer_id, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching in cursor:
        #                 time_stamps.append({"date_ref": date_ref,
        #                                     "val_scoreing_risk": f_check_none(val_scoring_risk),
        #                                     "class_scoring_risl": f_check_none(class_scoring_risk),
        #                                     "val_scoring_ai": f_check_none(val_scoring_ai),
        #                                     "class_scoring_ai": f_check_none(class_scoring_ai),
        #                                     "val_scoring_cr": f_check_none(val_scoring_cr),
        #                                     "class_scoring_cr": f_check_none(class_scoring_cr),
        #                                     "val_scoring_bi": f_check_none(val_scoring_bi),
        #                                     "class_scoring_bi": f_check_none(class_scoring_bi),
        #                                     "val_scoring_sd": f_check_none(val_scoring_sd),
        #                                     "class_scoring_sd": f_check_none(class_scoring_sd),
        #                                     "pre_notching": pre_notching
        #                                     })
        #             G.nodes[id]['risk_attribute'] = time_stamps
        #             G.nodes[id]['len_risk_attribute'] = len(time_stamps)
        #             counter_risk_id += 1
        #             if (counter_risk_id % 100) == 0:
        #                 print("{}/{}".format(counter_risk_id, len(risk_ids)))
        # pickle.dump(G, open("graph_only_risk_attribue.bin", "wb"))
        # print("Number of risk customer: {}".format(counter_risk_id))


        G = pickle.load(open("graph_only_risk_attribue.bin", "rb"))
        print("total nodes: {}".format(G.number_of_nodes()))
        print("total edges: {}".format(G.number_of_edges()))

        # remove node with no risk
        node_to_remove = []
        for node_id in G.nodes:
            if not 'risk_attribute' in G.nodes[node_id]:
                node_to_remove.append(node_id)
        G.remove_nodes_from(node_to_remove)
        print("total nodes: {}".format(G.number_of_nodes()))
        print("total edges: {}".format(G.number_of_edges()))
        # extract biggest connected component
        sub_graphs = sorted(nx.strongly_connected_component_subgraphs(G),
                            key=lambda sub_graph: sub_graph.number_of_nodes(), reverse=True)
        for cc_id, sub_graph in enumerate(sub_graphs):
            print("{}\t{}".format(cc_id, len(sub_graph)))
            if cc_id == 0:
                number_of_risky_users = 0
                print(sub_graph.number_of_nodes())
                print(sub_graph.number_of_edges())
                for node_id in sub_graph.nodes:
                    if 'risk_attribute' in sub_graph.nodes[node_id]:
                        number_of_risky_users += 1
                print("risk_attribute:{}".format(number_of_risky_users))
                # pickle.dump(sub_graph, open("cc_graph_only_risk_.bin", "wb"))
                break


        # cursor.execute(GET_ALL_CUSTOMER)
        # for customer_id, birth_date, b_partner, cod_uo, zipcode, region, country_code, customer_kind, kind_desc, customer_type, type_desc, uncollectable_status, ateco, sae in cursor:
        #     if G.has_node(customer_id):
        #         node_attribute = {
        #             "birth_date": birth_date,
        #             "b_partner": b_partner,
        #             "cod_uo": cod_uo,
        #             "zipcode": zipcode,
        #             "region": region,
        #             "country_code": country_code,
        #             "customer_kind": customer_kind,
        #             "kind_desc": kind_desc,
        #             "customer_type": customer_type,
        #             "type_desc": type_desc,
        #             "uncollectable_status": uncollectable_status,
        #             "ateco": ateco,
        #             "sae": sae
        #         }
        #         G.nodes[customer_id]['node_attribute'] = node_attribute
        #
        # pickle.dump(G, open("graph_node_attribute.bin", "wb"))

        # G = pickle.load(open("graph_risk_rev_attribue.bin", "rb"))
        #
        # cursor.execute(GET_ALL_ONEMANCOMPANY)
        # for customer_id, customer_id_join in cursor:
        #     G.add_edge(customer_id, customer_id_join, cod_type="1M", des_link_type="ONE_MAN_COMPANY")
        # print("added one man company")



        # # get all risky nodes
        # cursor.execute(GET_ALL_RISK_ID_ONEMANCOMPANY)
        # risk_ids = []
        # counter_risk_id = 0
        # for id, in cursor:
        #     risk_ids.append(id)
        # # add risk attribute
        # for id in risk_ids:
        #     if G.has_node(id):
        #         if 'risk_attribute' in G.nodes[id]:
        #             counter_risk_id += 1
        #         else:
        #             cursor.execute(GET_RISK_BY_CUSTOMER_ID.format(id))
        #             time_stamps = []
        #             for customer_id, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching in cursor:
        #                 time_stamps.append({"date_ref": date_ref,
        #                                     "val_scoreing_risk": f_check_none(val_scoring_risk),
        #                                     "class_scoring_risl": f_check_none(class_scoring_risk),
        #                                     "val_scoring_ai": f_check_none(val_scoring_ai),
        #                                     "class_scoring_ai": f_check_none(class_scoring_ai),
        #                                     "val_scoring_cr": f_check_none(val_scoring_cr),
        #                                     "class_scoring_cr": f_check_none(class_scoring_cr),
        #                                     "val_scoring_bi": f_check_none(val_scoring_bi),
        #                                     "class_scoring_bi": f_check_none(class_scoring_bi),
        #                                     "val_scoring_sd": f_check_none(val_scoring_sd),
        #                                     "class_scoring_sd": f_check_none(class_scoring_sd),
        #                                     "pre_notching": pre_notching
        #                                     })
        #             G.nodes[id]['risk_attribute'] = time_stamps
        #             G.nodes[id]['len_risk_attribute'] = len(time_stamps)
        #             counter_risk_id += 1
        #             if (counter_risk_id % 100) == 0:
        #                 print("{}/{}".format(counter_risk_id,len(risk_ids)))
        # pickle.dump(G, open("graph_omc_risk_attribue.bin", "wb"))
        # print("Number of risk customer: {}".format(counter_risk_id))
        #
        #
        #
        # # get all profitable nodes
        # cursor.execute(GET_ALL_REVENUE_ID_ONEMANCOMPANY)
        # revenue_id = []
        # counter_revenue_id = 0
        # for id, in cursor:
        #     revenue_id.append(id)
        #
        # # add revenue attribute
        # for id in revenue_id:
        #     if G.has_node(id):
        #         if 'revenue_attribute' in G.nodes[id]:
        #             counter_revenue_id += 1
        #         else:
        #             cursor.execute(GET_REVENUE_BY_CUSTOMER_ID.format(id))
        #             time_stamps = []
        #             for customerid, date_ref, val_scoring_rev, class_scoring_rev,  val_scoring_op, class_scoring_op,  val_scoring_co, class_scoring_co in cursor:
        #                 time_stamps.append({
        #                     "date_ref": date_ref,
        #                     "val_scoring_rev": f_check_none(val_scoring_rev),
        #                     "class_scoring_rev": f_check_none(class_scoring_rev),
        #                     "val_scoring_op": f_check_none(val_scoring_op),
        #                     "class_scoring_op": f_check_none(class_scoring_op),
        #                     "val_scoring_co": f_check_none(val_scoring_co),
        #                     "class_scoring_co": f_check_none(class_scoring_co)
        #                 })
        #             G.nodes[id]['revenue_attribute'] = time_stamps
        #             G.nodes[id]['len_revenue_attribute'] = len(time_stamps)
        #             counter_revenue_id += 1
        #             if (counter_revenue_id % 100) == 0:
        #                 print("{}/{}".format(counter_revenue_id,len(revenue_id)))
        #
        # print("Number of revenue customer: {}".format(counter_revenue_id))
        # pickle.dump(G, open("graph_omc_risk_rev_attribue.bin", "wb"))
    finally:
        cursor.close()
        cnx.close()


