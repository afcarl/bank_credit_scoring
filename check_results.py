from helper import CustomerDataset, get_embeddings, RiskToTensor, AttributeToTensor, ensure_dir
from datasets.sintetic.utils import get_sintetic_embeddings

from os.path import join as path_join
from torch.utils.data import DataLoader
from models import SimpleStructuredNeighborAttentionRNN
import torch.optim as optim
from torch.autograd import Variable
import torch
import random

import argparse
import visdom
from datetime import datetime
import pickle

BASE_DIR = "./data"
DATASET = "sintetic"
MODEL = "RNN_NetAttention_TimeAttention"



vis = visdom.Visdom()
EXP_NAME = "exp-{}".format(datetime.now())

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
            columnnames=list(map(str, range(weights.size(0)))),
            rownames=rowname,
            colormap=colorscale,
            marginleft=80
        ),
        win="win:check-{}-id{}-{}".format(EXP_NAME,id,title)
    )


if __name__ == "__main__":
    examples = pickle.load(open(path_join(BASE_DIR, DATASET, MODEL, "saved_eval_iter_2.bin"), "rb"))
    for example_id, example in examples.items():
        print("idx:{}\ttarget:{}\tpredicted:{}".format(example["id"], example["target"], example["predict"]))
        print("input:{}\nneighbors:{}".format(example["input"], example["neighbors"]))
        plot_heatmap(example["weights"].net_weight.sum(1), "net_weight", id=example_id)
        plot_heatmap(example["weights"].time_weight.sum(1), "time_weight", id=example_id)