import helper
import pandas as pd
import numpy as np
from itertools import tee
import torch
import visdom
from itertools import cycle
import os.path as path
import datetime as dt

vis = visdom.Visdom(port=8080)

TIMESTAMP = ['2016-01-31', '2016-02-29', '2016-03-31', '2016-04-30',
             '2016-05-31', '2016-06-30', '2016-07-31', '2016-08-31',
             '2016-09-30', '2016-10-31', '2016-11-30', '2016-12-31',
             '2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30',
             '2017-05-31', '2017-06-30']

BINS = np.array([0, 0.0845, 0.24, 0.4054, 0.575, 0.8848, 1.4213, 2.2963, 3.7337, 6.1263, 10.024, 17.3221, 31.6001, 100])
BASE_DIR = path.join("..", "..", "data", "customers")


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def draw_color_barchart(bar_plots, row_names, title, color):
    """
    draw bar chart with overlapping beans
    :param bar_plots:
    :param row_names:
    :param title:
    :param color:
    :return:
    """
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
    """
    draw overlapping lines in visdom
    :param X: X-value
    :param Y: Y-value
    :param legends: legend of the different lines
    :param line_types: line-types
    :param colors: color for each line
    :param title: plot tile
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :return:
    """


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

def histogram_by_class_by_timestamp(customers_data, time_step, attribute="class_scoring_risk"):
    assert attribute in customers_data.columns.level[1], "attribute not present"
    # select timestemp
    customers_data_time = customers_data.ix[time_step]

    histogram, _ = np.histogram(customer_data_array[:, 0], bins=BINS)
    print(histogram)

    # create histogram by class
    bar_plot = create_class_histogram(customer_data_array, BINS)
    print(bar_plot)


    row_names = ["{:.2f} {:.2f}".format(val_1, val_2) for val_1, val_2 in pairwise(BINS.tolist())]
    color = ['rgb(0,102,51)', 'rgb(51,255,51)', 'rgb(255,255,102)', 'rgb(255,255,0)',
             'rgb(255, 178, 102)', 'rgb(204,104,0)', 'rgb(102,255,255)', 'rgb(102,102,255)', 'rgb(0,0,204)',
             'rgb(204,153,255)', 'rgb(102,0,204)', 'rgb(255,153,153)', 'rgb(255, 0, 0)']
    draw_color_barchart(bar_plot, row_names, "Val-risk by with pre-notching. Date-{}".format(TIMESTAMP[timestamp_index]), color)



def create_class_histogram(data, bins):
    ret = np.zeros((bins.shape[0]-1, bins.shape[0]-1))
    for class_id in range(bins.shape[0]-1):
        class_index = np.where(data[:, 1] == class_id+1)
        histogram, bins_edge = np.histogram(data[:, 0][class_index], bins=bins)
        ret[class_id] = histogram
    return ret


if __name__ == "__main__":
    customers_data = pd.read_msgpack(path.join(path.join(BASE_DIR, "temp", "customers_risk_time_frame_null_df_final.msg")))
    time_step = dt.datetime.strptime(TIMESTAMP[0], "%Y-%m-%d")

    histogram_by_class_by_timestamp(customers_data[:, pd.IndexSlice[:, "pre_notching", "val_scoring_risk", "class_scoring_risk", "class_scoring_pre"]], time_step)
