import mysql.connector
import pickle
import helper as h
from collections import OrderedDict
import numpy as np
import visdom
from itertools import tee
from bidict import bidict
from itertools import cycle
from os.path import join as path_join
import torch
vis = visdom.Visdom()

config = {
  'user': 'root',
  'password': 'vela1990',
  'host': '127.0.0.1',
  'database': 'ml_crif',
}

TIMESTAMP = ["2016-06-30", "2016-07-31", "2016-08-31", "2016-09-30", "2016-10-31", "2016-11-30", "2016-12-31",
             "2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30", "2017-05-31", "2017-06-30"]

GET_ALL_CUSTOMER = "SELECT customerid FROM customers"
GET_ALL_OWNER = "SELECT customerid FROM onemancompany_owners"
CUSTOMERS_OWNER_UNION = "SELECT c.customerid FROM customers AS c UNION SELECT o.customerid FROM onemancompany_owners AS o"
GET_REVENUE_USER = "SELECT customerid FROM revenue"
GET_RISK_USER = "SELECT customerid, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching  FROM risk ORDER BY customerid asc, date_ref asc"
GET_RISK_USER_BY_ID = "SELECT customerid, date_ref, val_scoring_risk, class_scoring_risk, val_scoring_pre, class_scoring_pre, val_scoring_ai, class_scoring_ai, val_scoring_cr, class_scoring_cr, val_scoring_bi, class_scoring_bi, val_scoring_sd, class_scoring_sd, pre_notching  FROM risk ORDER BY date_ref asc WHERE customerid={}"
GET_ALL_CUSTOMER_LINKS_ID = "SELECT DISTINCT * FROM (SELECT c_one.customerid FROM customer_links AS c_one UNION SELECT c2.customerid_link FROM customer_links AS c2) AS u"
GET_ALL_CUSTOMER_LINKS_BY_ID = "SELECT DISTINCT customerid_link FROM customer_links WHERE customerid={}"
GET_ALL_RISK_LINKS_BY_CUSTOMERID = "SELECT DISTINCT cl.customerid, cl.customerid_link, cl.cod_link_type,  cl.des_link_type FROM risk AS r, customer_links AS cl WHERE r.customerid = cl.customerid AND r.customerid={}"
GET_DEFAULT_RISK_CUSTOMER = "SELECT r.customerid, r.date_ref, r.val_scoring_risk, r.class_scoring_risk, r.val_scoring_pre, r.class_scoring_pre, r.val_scoring_ai, r.class_scoring_ai, r.val_scoring_cr, r.class_scoring_cr, r.val_scoring_bi, r.class_scoring_bi, r.val_scoring_sd, r.class_scoring_sd, r.pre_notching  FROM risk AS r  WHERE r.customerid IN (SELECT DISTINCT r1.customerid FROM ml_crif.risk AS r1 WHERE r1.val_scoring_risk=100) ORDER BY r.customerid asc, r.date_ref asc"

GET_RISK_CUSTOMER_BY_ACCORDATO = "SELECT r.customerid FROM risk AS r LEFT OUTER JOIN features AS f ON r.bankid = f.bankid AND r.customerid = f.customerid AND r.date_ref = f.date_ref WHERE f.cod_feature = 'CR0018' AND r.val_scoring_risk = 100 ORDER BY f.value1 desc, r.customerid asc, r.date_ref desc LIMIT 1000"
GET_ACCORDATO_MASSIMO_BY_CUSTOMERID = "SELECT f.customerid, f.date_ref, f.value1, f.value2, f.cod_feature FROM features AS f WHERE f.cod_feature IN ('CR0018', 'CR0021', 'CR0040') AND f.customerid={}"

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor(buffered=True)

    
def recompute_diff_data(customer_data, customer_neigh_data):
    '''
    recompute relative and absolute difference between user risk and neighborhod risk
    :param customer_data:
    :param customer_neigh_data:
    :return:
    '''
    customer_diff_data = OrderedDict()
    customer_rel_diff_data = OrderedDict()
    for customer_id in sorted(customer_neigh_data.keys()):
        customer_risk = customer_data[customer_id]
        c_risk = np.array(customer_risk)
        c_r_risk = np.diff(c_risk, axis=0)
        for neiborhod_risk in customer_neigh_data[customer_id]:
            n_risk = np.array(neiborhod_risk)
            n_r_risk = np.diff(n_risk, axis=0)

            diff = np.fabs(c_risk - n_risk)
            if customer_id in customer_diff_data:
                customer_diff_data[customer_id].append(diff)
            else:
                customer_diff_data[customer_id] = [diff]

            # compute relative difference
            diff = np.fabs(np.array(c_r_risk) - np.array(n_r_risk))
            if customer_id in customer_rel_diff_data:
                customer_rel_diff_data[customer_id].append(diff)
            else:
                customer_rel_diff_data[customer_id] = [diff]

    return customer_diff_data, customer_rel_diff_data


def compute_difference_neighbor_mean(customer_diff_data):
    '''
    compute the mean of the user/neighbor difference
    :return: numpy array of the difference
    '''

    customer_diff_data_mean_by_neigh = OrderedDict()
    customer_diff_data_std_by_neigh = OrderedDict()

    for customer_id in customer_diff_data.keys():
        customer_diff_data_mean_by_neigh[customer_id] = np.mean(customer_diff_data[customer_id], axis=0)
        customer_diff_data_std_by_neigh[customer_id] = np.std(customer_diff_data[customer_id], axis=0)

    customer_mean_diff_value = np.array(list(customer_diff_data_mean_by_neigh.values()))
    customer_std_diff_value = np.array(list(customer_diff_data_std_by_neigh.values()))
    return customer_mean_diff_value, customer_std_diff_value


def compute_time_mean_difference(customer_diff_mean_neighbor_data):
    return np.mean(customer_diff_mean_neighbor_data, axis=0)


def draw_color_line(X, Y, legends, line_types, colors=['rgba(255, 178, 102, 1)',
                                 'rgba(204, 102, 0, 1)',
                                 'rgba(102, 255, 255, 1)',
                                 'rgba(0, 0, 255, 1)',
                                 'rgba(36, 255, 87, 1)',
                                 'rgba(255, 0, 0, 1)'  
                                 'rgba(67, 67, 67, 1)',
                                 'rgba(115, 115, 115, 1)',
                                 'rgba(102, 0, 204, 1)',
                                 'rgba(16, 111, 0, 1)',
                                  'rgba(102, 178, 255, 1)'], title='', xlabel='', ylabel=''):
    traces = []
    for ys, legend, line_type in zip(Y, legends, line_types):
        for row, (y, color) in enumerate(zip(ys, cycle(colors))):
            traces.append(dict(
                x=X,
                y=y.tolist(),
                mode='lines',
                type='line',
                line=dict(color=color, width=1, dash=line_type),
                connectgaps=True,
                name="{}-{}".format(legend, row)
            ))



    layout = dict(title=title,
                  xaxis=dict(title=xlabel,
                             showline=True,
                             showgrid=True,
                             showticklabels=True,
                             ticks='outside',
                             tickwidth=2,
                             ticklen=5,
                             nticks=len(X)
                             ),
                  yaxis=dict(title=ylabel,
                             showline=True,
                             showgrid=True,
                             showticklabels=True,
                             ticks='outside'),
                  width=800,
                  height=500,
                  showlegend=False
                  )

    return vis._send({'data': traces, 'layout': layout, 'win': 'win_{}'.format(title)})

def draw_color_barchart(bar_plots, row_names, title, color):
    data = [dict(x=row_names,
                 y=bar_plots[i,:].tolist(),
                 type='bar',
                 name="Class-{}".format(i+1),
                 marker=dict(color=color[i])
                 )
             for i in range(bar_plots.shape[0])]

    layout = dict(title=title,
                  barmode='stack',
                  width=700,
                  height=600,
                  showlegend=True
                  )
    vis._send({'data': data, 'layout': layout, 'win': "win_{}".format(title)})

def create_class_histogram(data, bins):
    ret = np.zeros((bins.shape[0]-1, bins.shape[0]-1))
    for class_id in range(bins.shape[0]-1):
        class_index = np.where(data[:, 1] == class_id+1)
        histogram, bins_edge = np.histogram(data[:, 0][class_index], bins=bins)
        ret[class_id] = histogram
    return ret

def histogram_by_class_by_timestamp(customer_data_array, timestamp_index=0):
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    # select timestemp
    customer_data_array = customer_data_array[:, timestamp_index, :]
    # remove nan
    customer_data_array = customer_data_array[~np.isnan(customer_data_array).any(axis=1)]

    bins = np.array([0, 0.0845, 0.24, 0.4054, 0.575, 0.8848, 1.4213, 2.2963, 3.7337, 6.1263, 10.024, 17.3221, 31.6001, 100])

    histogram, _ = np.histogram(customer_data_array[:, 0], bins=bins)
    print(histogram)

    # create histogram by class
    bar_plot = create_class_histogram(customer_data_array, bins)
    print(bar_plot)


    row_names = ["{:.2f} {:.2f}".format(val_1, val_2) for val_1, val_2 in pairwise(bins.tolist())]
    color = ['rgb(0,102,51)', 'rgb(51,255,51)', 'rgb(255,255,102)', 'rgb(255,255,0)',
             'rgb(255, 178, 102)', 'rgb(204,104,0)', 'rgb(102,255,255)', 'rgb(102,102,255)', 'rgb(0,0,204)',
             'rgb(204,153,255)', 'rgb(102,0,204)', 'rgb(255,153,153)', 'rgb(255, 0, 0)']
    draw_color_barchart(bar_plot, row_names, "Val-risk by with pre-notching. Date-{}".format(TIMESTAMP[timestamp_index]), color)

def cumulative_line_by_timestemap(customer_data_array, timestamp_index=0):
    # bins = np.array([-1, 0, 0.0845, 0.24, 0.41, 0.58, 0.89, 1.43, 2.30, 3.74, 6.13, 10.03, 17.33, 31.61, 100])
    bins = np.linspace(-1, 100, 1100)
    histogram, _ = np.histogram(customer_data_array[:, 0], bins=bins)
    vis.line(X=bins[10:],
             Y=np.cumsum(histogram)[9:],
             opts=dict(
                 fillarea=False,
                 showlegend=False,
                 width=600,
                 height=400,
                 xlabel='Risk value',
                 ylabel='# Customer',
                 # xtickmin=0,
                 # xtickmax=100,
                 # xtype='linear',
                 xtickmin=-2,
                 xtickmax=2,
                 xtype='log',
                 title='Only changing customers Risk value {}'.format(TIMESTAMP[timestamp_index+1]), )
             )

    # vis.line(X=bins[10:],
    #          Y=np.cumsum(histogram)[9:],
    #          opts=dict(
    #              fillarea=False,
    #              showlegend=False,
    #              width=600,
    #              height=400,
    #              xlabel='Difference risk value',
    #              ylabel='# Customer',
    #              xtype='log',
    #              title='Cumulative difference {}'.format(TIMESTAMP[timestamp_index + 1]), )
    #          )

def extract_numpy_data_by_selected_timestemp(customer_data):
    np_risk_all_customer_by_timestemp = np.empty((0, len(TIMESTAMP), 3))
    customer_to_id = bidict()

    for customer_id in sorted(customer_data.keys()):
        risk_customer = customer_data[customer_id]['risk_attribute']
        risk_data = []
        for timestemp in TIMESTAMP:
            try:
                risk_data.append(
                    [risk_customer[timestemp]['val_scoring_risk'], risk_customer[timestemp]['class_scoring_risk'], risk_customer[timestemp]['class_scoring_pre']])
            except KeyError as ke:
                risk_data.append([np.nan, np.nan, np.nan])
                continue

        np_risk_all_customer_by_timestemp = np.concatenate((np_risk_all_customer_by_timestemp, np.array([risk_data])),axis=0)
        customer_to_id[customer_id] = len(customer_to_id)

    pickle.dump(np_risk_all_customer_by_timestemp, open(
        path_join("./data", "customers", "np_risk_customers_by_selected_timestemp.bin"), "wb"))
    pickle.dump(customer_to_id, open(path_join("./data", "customers", "customer_to_id.bin"), "wb"))

def customer_neighbor_difference():
    customer_data = pickle.load(open("new_risk_customers_with_risk_neighbor.bin", "rb"))
    np_risk_all_customer_by_timestemp = pickle.load(open("new_np_risk_customers_by_selected_timestemp.bin", "rb"))
    customer_to_id = pickle.load(open("new_customer_to_id.bin", "rb"))
    np_risk_all_risk_by_timestamp_wrt_neighbor = np.empty((0, 3, 2))
    count = 0
    for customer_id in sorted(customer_to_id.keys()):
        customer_idx = customer_to_id[customer_id]
        try:
            neighbors = customer_data[customer_id]['risk_neighbor']
            for neighbor_id in neighbors:
                try:
                    neighbor_idx = customer_to_id[neighbor_id]
                    np_risk_all_risk_by_timestamp_wrt_neighbor = np.concatenate(
                        (np_risk_all_risk_by_timestamp_wrt_neighbor,
                         np.abs(
                             [np_risk_all_customer_by_timestemp[customer_idx][1:, :] -
                              np_risk_all_customer_by_timestemp[neighbor_idx][1:, :]])
                         ), axis=0)

                except KeyError as ke:
                    # print("{}\t{}".format(customer_id, neighbor_id))
                    continue
        except KeyError as ke:
            print("----{}\t{}".format(customer_id, ke))
            continue
        count += 1
        if count % 100 == 0:
            print(count)

    pickle.dump(np_risk_all_risk_by_timestamp_wrt_neighbor,
                open("new_np_risk_all_risk_by_timestamp_wrt_neighbor.bin", "wb"))

def plot_customer_default_timeseries(np_customer_data):
    # np_default_customers = np_customer_data[~np.isnan(np_customer_data).any(axis=1).any(axis=1)]
    draw_color_line(X=TIMESTAMP, Y=[np_customer_data[:, :, 2].numpy(), np_customer_data[:, :, -2].numpy()],
                    legends=["risk", "accordato"],
                    line_types=["", "dash"],
                    title="Risk csvs for eventually default customers",
                    xlabel='Timestemp',
                    ylabel='Risk value'
                    )
    print(np_customer_data[:100, :])


f_parse_date = lambda x: "{}-{}-{}".format(x[:4], x[4:6], x[6:])
if __name__ == "__main__":
    try:
        # risky_customer = set()
        # LIMIT = 10
        # cursor.execute(GET_RISK_CUSTOMER_BY_ACCORDATO)
        # for customer_id, in cursor:
        #     risky_customer.add(customer_id)
        #     if len(risky_customer) == LIMIT:
        #         break
        #
        # customer_id_2_customer_idx = pickle.load(open(path_join(".", "data", "customers", "customerid_to_idx.bin"), "rb"))

        risky_customer = pickle.load(open("data/customers/temp/risky_customer_10.bin", "rb"))
        # accordato_embedding = torch.FloatTensor(len(risky_customer), len(TIMESTAMP), 6).zero_()
        # for idx, c_id in enumerate(sorted(risky_customer)):
        #     cursor.execute(GET_ACCORDATO_MASSIMO_BY_CUSTOMERID.format(c_id))
        #     c_accordato = torch.FloatTensor(len(TIMESTAMP), 6).zero_()
        #     for customer_id, date_ref, value1, value2, cod_feature in cursor:
        #         if cod_feature == 'CR0021':
        #             c_accordato[TIMESTAMP.index(f_parse_date(date_ref)), 2:4] = torch.FloatTensor([value1, value2])
        #         elif cod_feature == 'CR0040':
        #             c_accordato[TIMESTAMP.index(f_parse_date(date_ref)), 4:] = torch.FloatTensor([value1, value2])
        #         else:
        #             c_accordato[TIMESTAMP.index(f_parse_date(date_ref)), :2] = torch.FloatTensor([value1, value2])
        #     accordato_embedding[idx] = c_accordato
        #     print(idx)
        #
        #
        # pickle.dump(accordato_embedding, open("data/customers/temp/accordato_embedding.bin","wb"))
        # pickle.dump(risky_customer, open("data/customers/temp/risky_customer_10.bin", "wb"))

        customer_id_2_customer_idx = pickle.load(open(path_join(".", "data", "customers", "customerid_to_idx.bin"), "rb"))

        accordato_embedding = pickle.load(open("data/customers/temp/accordato_embedding.bin", "rb"))

        risk_tsfm = h.RiskToTensor(path_join(".", "data", "customers"))
        attribute_tsfm = h.AttributeToTensor(path_join(".", "data", "customers"))
        input_embeddings, target_embeddings, neighbor_embeddings, seq_len = h.get_embeddings(path_join(".", "data", "customers"),
                                                                                             "customers_formatted_attribute_risk.bin",
                                                                                             "customeridx_to_neighborsidx.bin",
                                                                                             24,
                                                                                             risk_tsfm,
                                                                                             attribute_tsfm)
        customer_idx = [customer_id_2_customer_idx[c_id] for c_id in sorted(risky_customer)]
        plot_customer_default_timeseries(torch.cat((input_embeddings[customer_idx], torch.log(accordato_embedding+1)), dim=2))

        # customer_data = pickle.load(open(path_join("./data", "customers", "customers_risk.bin"), "rb"))
        # extract_numpy_data_by_selected_timestemp(customer_data)
        # np_customer_data = pickle.load(open("np_risk_customers_by_selected_timestemp.bin", "rb"))
        # customer_to_id = pickle.load(open("customer_to_id.bin", "rb"))
        # risky_customer_idx = np.array([customer_to_id[customer_id] for customer_id in risky_customer])
        #
        # plot_customer_default_timeseries(np_customer_data[risky_customer_idx])













    finally:
        cursor.close()
        cnx.close()