from os.path import join as path_join
import visdom
import datetime
import pickle
import plotly.graph_objs as go
import torch
import collections
import plotly.plotly as py
import plotly.tools as tls
import copy

BASE_DIR = "./data"
DATASET = "sintetic"
MODEL = "RNN_TimeAttention"



vis = visdom.Visdom()
EXP_NAME = "exp-{}".format(datetime.datetime.now())

def _axisformat(x, opts):
    fields = ['type', 'tick', 'label', 'tickvals', 'ticklabels', 'tickmin', 'tickmax', 'tickfont']
    if any([opts.get(x + i) for i in fields]):
        return {
            'type': opts.get(x + 'type'),
            'title': opts.get(x + 'label'),
            'range': [opts.get(x + 'tickmin'), opts.get(x + 'tickmax')]
            if (opts.get(x + 'tickmin') and opts.get(x + 'tickmax')) is not None else None,
            'tickvals': opts.get(x + 'tickvals'),
            'ticktext': opts.get(x + 'ticklabels'),
            'tickwidth': opts.get(x + 'tickstep'),
            'showticklabels': opts.get(x + 'ytick'),
        }


def __reformat_duplicates__(a):
    counter = collections.Counter(a)

    for item, count in counter.items():
        if count > 1:
            c = 0
            for idx, val in enumerate(a):
                if val == item:
                    a[idx] = val + str(c)
                    c += 1

    return a


def plot_time_attention(weights, data, title, id, colorscale="Viridis"):
    # programmers = ['Alex', 'Nicole', 'Sara', 'Etienne', 'Chelsea', 'Jody', 'Marianne']
    #
    # base = datetime.datetime.today()
    # date_list = [base - datetime.timedelta(days=x) for x in range(0, 180)]
    #
    # z = []
    #
    # for prgmr in programmers:
    #     new_row = []
    #     for date in date_list:
    #         new_row.append(np.random.poisson())
    #     z.append(list(new_row))
    #
    # data = [
    #     go.Heatmap(
    #         z=z,
    #         x=date_list,
    #         y=programmers,
    #         colorscale='Viridis',
    #     )
    # ]



    for net in range(weights.size(0)):
        fn_y_name = lambda x: "Node Input" if x==0 else "Neighbor {}".format(x)
        row_name = list(map(lambda x: str(x)[:3], data[0].numpy().tolist()))
        row_val = __reformat_duplicates__(copy.copy(row_name))
        col_name = list(map(lambda x: str(x)[:3], data[net].numpy().tolist()))
        col_val = __reformat_duplicates__(copy.copy(col_name))
        plot_data = [
                go.Heatmap(
                    z=weights[net].numpy().tolist(),
                    x=row_val,
                    y=col_val,
                    colorscale=colorscale,
                )
            ]
        layout = go.Layout(
            title='{}-{}'.format(title, net),
            xaxis=dict(
                type="category",
                tickmode="array",
                tickvals=row_val,
                ticktext=row_name,
                autotick=False,
                title='Node Input',
                showticklabels=True,
                tickangle=0,
                showgrid=True,
                mirror='ticks',
            ),
            yaxis=dict(
                type="category",
                tickmode="array",
                tickvals=col_val,
                ticktext=col_name,
                autotick=False,
                title=fn_y_name(net),
                showticklabels=True,
                tickangle=0,
                showgrid=True,
                mirror='ticks',
            ),
        )

        fig = go.Figure(data=plot_data, layout=layout)
        py.plot(fig, filename='{}-{}-{}'.format(id, title, net))

        #
        # vis.heatmap(
        #     X=weights[net],
        #     opts=dict(
        #         title=title,
        #         columnnames=col_name,
        #         rownames=row_name,
        #         colormap=colorscale,
        #         marginleft=80
        #     ),
        #     win="win:check-{}-id{}-{}".format(EXP_NAME, net, title)
        # )


def plot_heatmap(weights, title, id=0, colorscale="Viridis"):
    weights_norm = weights.div(weights.max(dim=1)[0].unsqueeze(1))
    if weights.size(1) == 4:
        weights_norm = weights_norm.t()
        rowname = ["neighbor {}".format(i) for i in range(1, 5)]
    else:
        rowname = ["node"]
        rowname.extend(["neighbor {}".format(i) for i in range(1, 5)])

    # traces = [dict(
    #         z=[weights_norm[row, :].numpy().tolist()],
    #         x=list(map(lambda x: str(int(x)), data[row, :])),
    #         y=[str(row)],
    #         zmin=z_mins[row],
    #         zmax=z_maxs[row],
    #         type='heatmap',
    #         colorscale=colorscale,
    #         xaxis='x{}'.format(row+1),
    #         yaxis='y{}'.format(row+1)
    #     ) for row in range(4)]
    # y_limits = [[0, 0.24], [0.25, 0.49], [0.5, 0.74], [0.75, 1]]
    # layout = dict(
    #         title='title',
    #         xaxis1=dict(domain=[0, 1]),
    #         yaxis1=dict(domain=[0, 0.24]),
    #     xaxis2=dict(domain=[0, 1]),
    #     yaxis2=dict(domain=[0.25, 0.49]),
    #     xaxis3=dict(domain=[0, 1]),
    #     yaxis3=dict(domain=[0.5, 0.74]),
    #     xaxis4=dict(domain=[0, 1]),
    #     yaxis4=dict(domain=[0.75, 1]))
    #
    # return vis._send({
    #     'data': traces,
    #     'layout': layout,
    #     'win': "win:check-{}".format(EXP_NAME),
    # })
    return vis.heatmap(
        X=weights_norm,
        opts=dict(
            title=title,
            columnnames=list(map(str, range(weights_norm.size(1)))),
            rownames=rowname,
            colormap=colorscale,
            marginleft=80
        ),
        win="win:check-{}-id{}-{}".format(EXP_NAME,id,title)
    )


if __name__ == "__main__":
    examples = pickle.load(open(path_join(BASE_DIR, DATASET, MODEL, "saved_eval_iter_10.bin"), "rb"))
    for example_id, example in examples.items():
        print("idx:{}\ttarget:{}\tpredicted:{}".format(example["id"], example["target"], example["predict"]))
        print("input:{}\nneighbors:{}".format(example["input"], example["neighbors"]))
        # plot_heatmap(example["weights"].sum(1), "net_weight", id=example_id)
        plot_time_attention(example["weights"], torch.cat((example["input"], example["neighbors"]), dim=0), "time_weight", id=example_id)