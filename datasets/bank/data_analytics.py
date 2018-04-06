import helper
import pandas as pd
import numpy as np
from itertools import tee
import plotly.plotly as py
import plotly.graph_objs as go
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

BINS = np.array([0, 0.0845, 0.23, 0.4054, 0.575, 0.8848, 1.4213, 2.2963, 3.7337, 6.1263, 10.024, 17.3221, 31.6001, 100])
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
    data = [go.Bar(
        x=row_names,
        y=bar_plots[i,:].tolist(),
        name="Class-{}".format(i + 1),
        marker=dict(color=color[i])) for i in range(bar_plots.shape[0])]

    layout = go.Layout(
        title=title.split("/")[1],
        yaxis=dict(
            title='Customer number',
            autorange=False,
            showgrid=True,
            zeroline=True,
            showline=True,
            range=[0, 2500],
            zerolinecolor='#000000',
            zerolinewidth=2,
            tickfont=dict(
                size=20,
                color='black')
        ),
        xaxis=dict(
            title='Risk value',
            zerolinecolor='#000000',
            zerolinewidth=2,
            tickfont=dict(
                size=20,
                color='black')
        ),
        barmode='stack',
        width=1200,
        height=600,
        showlegend=True)
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=title)


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

def __pd_histogram__(df, _idx, class_labels=range(1, 14)):
    """
    compute the hisgotram for the current class using pandas.
    compute both the number of element as well the ids of each beans
    :param df: 
    :param _idx: 
    :param class_labels: 
    :return: 
    """
    class_beans_label = pd.cut(df[:, "val_scoring_risk"].iloc[_idx], bins=BINS, labels=class_labels)
    class_beans_label = class_beans_label.groupby(class_beans_label)
    class_histogram = class_beans_label.apply(lambda x: x.size).values
    class_id_histogram = class_beans_label.groups
    return class_histogram, class_id_histogram

def __customer_histogram__(data, attribute):
    """
    create the histogram base on the risk attribute given.
    :param data: 
    :param attribute: 
    :return: customers histogram, customers groupby ids
    """
    histogram = np.zeros((BINS.size - 1, BINS.size - 1))
    customers_id_histogram = {}
    for class_id in range(BINS.size - 1):
        class_index, = np.where(data[:, attribute] == class_id + 1)
        class_histogram, class_customers_id_histogram = __pd_histogram__(data, class_index)
        histogram[class_id] = class_histogram
        customers_id_histogram[class_id + 1] = class_customers_id_histogram
    return histogram, customers_id_histogram

def histogram_by_class_by_timestamp(customers_data, attribute="class_scoring_pre", print_anomalies=True):


    # select timestemp
    assert attribute in customers_data.index.get_level_values("attribute").unique(), "attribute: {} not present in \n {}".format(attribute, customers_data.index.get_level_values("attribute").unique())

    histogram, _ = np.histogram(customers_data[:, "val_scoring_risk"], bins=BINS)
    print(histogram)

    # create histogram by class
    customers_histogram, customers_id_histogram = __customer_histogram__(customers_data, attribute)
    np.set_printoptions(suppress=True, linewidth=100)
    print(customers_histogram)


    row_names = ["{:.2f} {:.2f}".format(val_1, val_2) for val_1, val_2 in pairwise(BINS.tolist())]
    color = ['rgb(0,102,51)', 'rgb(51,255,51)', 'rgb(255,255,102)', 'rgb(255,255,0)',
             'rgb(255, 178, 102)', 'rgb(204,104,0)', 'rgb(102,255,255)', 'rgb(102,102,255)', 'rgb(0,0,204)',
             'rgb(204,153,255)', 'rgb(102,0,204)', 'rgb(255,153,153)', 'rgb(255, 0, 0)']

    if attribute == "class_scoring_risk":
        title = "Val-risk by with pre-notching/Date-{}".format(time_step.strftime('%d-%m-%Y'))
    else:
        title = "Val-risk by with-OUT pre-notching/Date-{}".format(time_step.strftime('%d-%m-%Y'))
    draw_color_barchart(customers_histogram, row_names, title, color)

    if print_anomalies:
        classes_ids = sorted(list(customers_id_histogram.keys()))
        for i in classes_ids:
            print("\n\n------------------\nanomalies in class: {}\n------------------".format(i))
            for j in classes_ids:
                if i == j:
                    continue
                if customers_id_histogram[i][j].size > 0:
                    print("moved to class: {}\t{}".format(j, customers_id_histogram[i][j].tolist()))







if __name__ == "__main__":
    customers_data = pd.read_msgpack(path.join(path.join(BASE_DIR, "temp", "customers_risk_time_frame_null_df_final.msg")))

    # for timestamp in TIMESTAMP:
    #     time_step = dt.datetime.strptime(timestamp, "%Y-%m-%d")
    #     histogram_by_class_by_timestamp(customers_data.loc[time_step, pd.IndexSlice[:, ["pre_notching", "val_scoring_risk", "class_scoring_risk", "class_scoring_pre"]]])
