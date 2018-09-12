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
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=200)
torch.set_printoptions(precision=6)
BASE_DIR = "../../data"
DATASET = "customers"
MODEL = "Jordan_RNN_FeatureTransformerAttention"


Axes_data = collections.namedtuple("Axes_data", "val name")


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
                    a[idx] = val + "." + str(c)
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




if __name__ == "__main__":
    examples = pickle.load(open(path_join(BASE_DIR, DATASET, MODEL, "saved_test_adam_temp-0.45.bin"), "rb"))
    customers_id_to_idx = pickle.load(open(path_join(BASE_DIR, DATASET, "customers_id_to_idx.bin"), "rb"))

    for example_id, example in examples.items():
        print("idx:{}\ttarget:{}\tpredicted:{}".format(example["id"], example["target"], example["predict"]))
        print("input:{}\nneighbors:{}".format(example["input"][:, 0], example["neighbors"][:, :, 0].t()))
        num_neighbors, time_steps, n_feature = example["neighbors"].size()
        edge_type = example["edge_type"][:, 0].numpy()
        values = []
        temp = example["neighbors"][:, :, 0].contiguous().view(-1).numpy().tolist()
        for i in range(num_neighbors):
            for j in range(time_steps):
                values.append(int(temp[(i * time_steps) + j]) + (0.01 * ((i * time_steps) + j)))

        # plot_time_attention((example["node_edge_interaction"]/2) / (example["node_edge_interaction"]/2).sum(dim=-1, keepdim=True).expand(-1, 30),
        #                     torch.cat((example["input"].t(), example["neighbors"]), dim=0), "time_weight", id=example_id)

        example["neigh_neigh_attention"] = (example["neigh_neigh_attention"] / 4) / (example["neigh_neigh_attention"] / 4).sum(dim=-1,
                                                                                                                            keepdim=True).expand(
            -1, num_neighbors * time_steps)
        row = Axes_data(val=list(range(time_steps)), name=list(map(lambda x: str(x), range(time_steps))))
        col = Axes_data(val=values, name=list(map(lambda v: str(v)[:5], values)))
        plot_time_attention(example["neigh_neigh_attention"], row, col, "neigh_neigh_interaction", id=example_id)

        example["node_neigh_attention"] = (example["node_neigh_attention"] / 4) / (example["node_neigh_attention"] / 4).sum(dim=-1,
                                                                                                                            keepdim=True).expand(
            -1, 2 * time_steps)
        time_steps, node_neigh_size = example["node_neigh_attention"].size()
        col = Axes_data(val=[i + 0.001 * j for i in range(2) for j in range(time_steps)],
                        name=list(map(lambda x: "{}".format(str(x)[:5]),
                                      [i + 0.001 * j for i in range(2) for j in range(time_steps)])))
        plot_time_attention(example["node_neigh_attention"], row, col, "node_neigh_interation", id=example_id)
