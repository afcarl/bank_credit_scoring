from os import path
import pandas as pd
from numpy.random import random_integers
import pickle
from requests import session, ConnectionError
from helper import ensure_dir
from bs4 import BeautifulSoup
import time

BASE_DIR = path.join("..", "..", "data", "pems")
START = "2017-05-01"
END = "2017-07-01"
BASE_URL = "http://pems.dot.ca.gov/?district_id=4&station_id={}&dnode=VDS"

UserName = "sandro001@e.ntu.edu.sg"
Pass = "Hwpenny(1"




def downloadFile(stations, authencication_data, url_base='http://pems.dot.ca.gov/'):
    with session() as c:
        c.post(url_base, data=authencication_data)

        for i, station_id in enumerate(stations):
            ts = 10  # Default time to sleep
            print("Iteration: " + str(i))
            print('initial time to sleep ' + str(ts))
            while True:
                try:  # Download with 10-second sleep time breaks
                    print('Downloading file number: ' + str(i))
                    print('try to download file: ' + str(i))
                    print('time to sleep ' + str(ts))
                    # Make the request and download attached file
                    link = BASE_URL.format(station_id)
                    r = c.get(link)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    for id, change_log in enumerate(filter(lambda x: x.contents[1].text.strip() == "Change Log", soup.find_all("table"))):
                        if id > 0:
                            raise Exception("Multiple change log for station: {}".format(station_id))
                        yield (station_id, [tag.get_text() for tag in change_log.find_all("td", style="white-space: nowrap")[-2:]])
                    time.sleep(random_integers(ts, int(1.2 * ts)))
                except ConnectionError:
                    print('ConnectionError')
                    ts = ts * 2
                    time.sleep(ts)  # Sleep and login again
                    c.post(url_base, data=dt)  # , params=p)
                    continue
                break




if __name__ == "__main__":
    stations = pd.read_csv(path.join(BASE_DIR, "stations.csv"), index_col="ID")
    dt = {
        'action': 'login',
        'username': UserName,
        'password': Pass
    }
    add_data = {}
    for station_id, (lat, lng) in downloadFile(stations.index, dt):
        try:
            add_data[station_id] = dict(Latitude=float(lat),
                                        Longitude=float(lng))
            print("Station {} done!".format(station_id))
        except Exception as e:
            print(e)
            print("\n\n{}".format(station_id))

    down_data = pd.DataFrame.from_dict(add_data, orient="index")
    down_data.to_csv(path.join(BASE_DIR, "lat_lng.csv"))





