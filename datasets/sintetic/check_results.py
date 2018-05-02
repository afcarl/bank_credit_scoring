from os.path import join as path_join
import visdom
import datetime
import pickle
import plotly.graph_objs as go
import numpy as np
import collections
import plotly.plotly as py
import plotly.tools as tls
import copy
from functools import reduce
BASE_DIR = "../../data"
DATASET = "sintetic"
MODEL = "RNN_TransformerAttention"

Axes_data = collections.namedtuple("Axes_data", "val name")


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

fn_flatten = lambda l: [item for sublist in l for item in sublist]
fn_y_name = lambda x: "Node" if x==0 else "Neighbor {}".format(x)

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


def plot_time_attention(weights, row_data, col_data, title, id, colorscale="Viridis"):
    # row_name = list(map(lambda x: str(x)[:3], data[0].numpy().tolist()))
    # row_val = __reformat_duplicates__(copy.copy(row_name))
    # col_name = fn_flatten([list(map(lambda x: str(x)[:3], data[i].numpy().tolist())) for i in range(data.size(0))])
    # col_val = __reformat_duplicates__(copy.copy(col_name))
    # print(len(col_val))


    plot_data = [
        go.Heatmap(
            z=weights,
            x=col_data.val,
            y=row_data.val,
            colorscale=colorscale,
        )
    ]
    layout = go.Layout(
        title='{}-{}'.format(title, id),
        xaxis=dict(
            type="category",
            tickmode="array",
            tickvals=col_data.val,
            ticktext=col_data.name,
            autotick=False,
            title='Neighbors',
            showticklabels=True,
            tickangle=0,
            showgrid=True,
            mirror='ticks',
        ),
        yaxis=dict(
            type="category",
            tickmode="array",
            tickvals=row_data.val,
            ticktext=row_data.name,
            title='Node input',
            autotick=False,
            showticklabels=True,
            tickangle=0,
            showgrid=True,
            mirror='ticks',
        ),
    )

    fig = go.Figure(data=plot_data, layout=layout)
    py.plot(fig, filename='{}-{}'.format(title, id))



def plot_heatmap(weights, title, id=0, colorscale="Viridis"):
    weights_norm = weights.div(weights.max(dim=1)[0].unsqueeze(1))
    if weights.size(1) == 4:
        weights_norm = weights_norm.t()
        rowname = ["neighbor {}".format(i) for i in range(1, 5)]
    else:
        rowname = ["node"]
        rowname.extend(["neighbor {}".format(i) for i in range(1, 5)])

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

    exp_ids = [2978, 8058]
    for example_id, example in examples.items():
        if example_id not in exp_ids:
            continue

        print("idx:{}\ttarget:{}\tpredicted:{}".format(example["id"], example["target"], example["predict"]))
        print("input:{}\nneighbors:{}".format(example["input"], example["neighbors"].t()))
        num_neighbors, time_steps, edge_types, _ = example["infer_edge_type"].size()


        # plot_time_attention((example["node_edge_interaction"]/2) / (example["node_edge_interaction"]/2).sum(dim=-1, keepdim=True).expand(-1, 30),
        #                     torch.cat((example["input"].t(), example["neighbors"]), dim=0), "time_weight", id=example_id)

        example["node_edge_interaction"] = (example["node_edge_interaction"]/4) / (example["node_edge_interaction"]/2).sum(dim=-1, keepdim=True).expand(-1, 30)
        row = Axes_data(val=list(range(time_steps)), name=list(map(lambda x: str(x), range(time_steps))))
        col = Axes_data(val=np.arange(1., edge_types + 2., 0.1).tolist(), name=list(map(lambda x: str(x)[:3], np.arange(1., edge_types + 2., 0.1).tolist())))
        plot_time_attention(example["node_edge_interaction"], row, col, "edge_interaction", id=example_id)

        row = Axes_data(val=list(range(time_steps)), name=list(map(lambda x: str(x), range(time_steps))))

        col = Axes_data(val=[i + 0.1*j for i in range(0, num_neighbors) for j in [1, 2]],
                        name=list(map(lambda x: "neigh_{}.{}".format(str(x+1)[0], str(x)[2]), [i + 0.1*j for i in range(0, num_neighbors) for j in [1, 2]])))
        edge_classification = example["infer_edge_type"].transpose(0, 1).contiguous().view(time_steps, -1)
        edge_classification[-1, -2:] = 1 - edge_classification[-1, -2:]
        plot_time_attention(edge_classification, row, col, "edge_type", id=example_id)
