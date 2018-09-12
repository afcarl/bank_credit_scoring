import torch
from os import path
from datasets.utils import BASE_DIR
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import networkx as nx

mapbox_access_token = "pk.eyJ1IjoiYW5kb21wZXN0YSIsImEiOiJjamhpcndldDcyNmQwMzZuZ2g3djRxa2NjIn0.RLop0vLZtQmf31qPCDpcHQ"

colorscale = [
    'rgb(68.0, 1.0, 84.0)',
    'rgb(66.0, 64.0, 134.0)',
    'rgb(38.0, 130.0, 142.0)',
    'rgb(63.0, 188.0, 115.0)',
    'rgb(216.0, 226.0, 25.0)'
]


def draw_graph(G, stations):
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='top center',
        textfont=dict(
            family='arial',
            size=3,
            color='rgb(0,0,0)'
        ),
        hoverinfo='none',
        marker=go.Marker(
            showscale=False,
            color='rgb(200,0,0)',
            opacity=0.3,
            size=8,
            line=go.Line(width=1, color='rgb(0,0,0)')))

    node_positions = {}
    for id in stations.index:
        lat, lng = stations.loc[id, ["Latitude", "Longitude"]].values
        node_positions[id] = [lat, lng]
        node_trace['x'].append(lat)
        node_trace['y'].append(lng)
        # node_trace['text'].append(id)

    # The edges will be drawn as lines:
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=2, color='rgb(10,10,10)'),
        hoverinfo='none',
        mode='lines')

    for id in node_positions.keys():
        for neigh_info in G.neighbors(id):
            x0, y0 = node_positions[id]
            x1, y1 = node_positions[neigh_info[0]]
            edge_trace['x'].extend([x0, x1, None])
            edge_trace['y'].extend([y0, y1, None])

    # Create figure:
    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                    layout=go.Layout(
                        title='Sample Network',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    py.plot(fig)

if __name__ == "__main__":
    G = torch.load(path.join(BASE_DIR, "pems", "temp", "graph.pt"))
    print(nx.average_degree_connectivity(G, source="in", target="out"))
    stations = pd.read_csv(path.join(BASE_DIR, "pems", "station_comp.csv"), index_col="ID")
    draw_graph(G, stations)

    #
    #
    # nx.draw(G)
    #
    # stations = pd.read_csv(path.join(BASE_DIR, "pems", "station_comp.csv"), index_col="ID")
    # lats = []
    # lngs = []
    # ids = []
    #
    # for id in stations.index:
    #     lat, lng = stations.loc[id, ["Latitude", "Longitude"]].values
    #     lats.append(str(lat)[:8])
    #     lngs.append(str(lng)[:8])
    #     ids.append(str(id))
    #
    # data = [
    #     go.Scattermapbox(
    #         lat=lats,
    #         lon=lngs,
    #         mode='markers',
    #         marker=dict(
    #             size=3
    #         ),
    #         text=ids,
    #     )
    # ]
    #
    # layout = go.Layout(
    #     autosize=True,
    #     hovermode='closest',
    #     mapbox=dict(
    #         accesstoken=mapbox_access_token,
    #         bearing=0,
    #         pitch=0,
    #         center=dict(
    #             lat=37.62,
    #             lon=-122.12
    #         ),
    #         zoom=5
    #     ),
    # )
    #
    # fig = dict(data=data, layout=layout)
    # py.plot(fig, filename='Stations')




