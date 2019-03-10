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
    stations_id_2_exp_id = torch.load(path.join(BASE_DIR, "pems", "station_id_to_exp_idx.pt"))
    exp_id = 34546
    station_id = stations_id_2_exp_id.inverse[exp_id]
    print(station_id)

    G = torch.load(path.join(BASE_DIR, "pems", "temp", "graph.pt"))
    print(nx.average_degree_connectivity(G, source="in", target="out"))


    neighbors_ids = G.neighbors(station_id[0])

    stations = pd.read_csv(path.join(BASE_DIR, "pems", "station_comp.csv"), index_col="ID")
    stations_distance = torch.load(path.join(BASE_DIR, "pems", "temp", "stations_distances.pt"))
    # draw_graph(G, stations)

    closest_stations = sorted(stations_distance[station_id[0]].items(), key=lambda x: x[1])
    closest_stations = list(filter(lambda x: x[1] > 0, closest_stations))
    print(closest_stations[:20])

    # nx.draw(G)
    # stations = pd.read_csv(path.join(BASE_DIR, "pems", "station_comp.csv"), index_col="ID")

    ids = [400996, 402930, 402931, 401891, 401890, 400440, 400668]
    # for id in ids:
    #     print("{}_{}".format(id, stations.loc[id, ["Latitude", "Longitude"]].values))

    lats = ['37.350929', '37.350929', '37.350929', '37.343293', '37.343447', '37.353727', '37.35512']
    lngs = ['-121.862878', '-121.862379', '-121.862379', '-121.855795', '-121.855539', '-121.865827', '-121.866967']

    weight = {
        (400996): 0.04,
        (402930): 1.81,
        (402931): 1.44,
        (401891): 0.03,
        (401890): 0.05,
        (400440): 0.03,
        (400668): 0.04
    }
    # lats = []
    # lngs = []
    # ids = []

    # for id in map(lambda x: x[0], closest_stations[:20]):
    #     lat, lng = stations.loc[id, ["Latitude", "Longitude"]].values
    #     lats.append(str(lat))
    #     lngs.append(str(lng))
    #     ids.append(str(id))


    # for id in stations.index:
    # for id in ids:
    #     lat, lng = stations.loc[id, ["Latitude", "Longitude"]].values
    #     lats.append(str(lat))
    #     lngs.append(str(lng))
        # ids.append(str(id))

    data = [
        go.Scattermapbox(
            lat=['37.348392'],
            lon=['-121.860589'],
            mode='markers',
            marker=dict(
                size=15,
                color='rgba(255, 51, 51, 1)'
            ),
            text=station_id,
            name="node"
        ),

        go.Scattermapbox(
            lat=lats,
            lon=lngs,
            mode='markers',
            marker=dict(
                size=10,
                color='rgba(51, 51, 255, 1)'
            ),
            text=ids,
            name="neighbours"
        )
    ]

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            pitch=0,
            center=dict(
                lat=37.348392,
                lon=-121.860589
            ),
            zoom=15
        ),
    )

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='Stations')




