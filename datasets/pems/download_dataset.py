from os import path
import pandas as pd
import numpy as np
import pickle
from requests import session, ConnectionError
from helper import ensure_dir
from bs4 import BeautifulSoup
import time
from datetime import datetime

BASE_DIR = path.join("..", "..", "data", "pems")
START = "2017-05-01"
END = "2017-07-01"
BASE_URL = "http://pems.dot.ca.gov/?district_id=4&station_id={}&dnode=VDS"

UserName = "sandro001@e.ntu.edu.sg"
Pass = "Hwpenny(1"


str_to_timestemp_fn = lambda x: time.mktime(datetime.strptime(x, "%m/%d/%Y %H:%M").timetuple())

def crawlLatLng(stations, authencication_data, url_base='http://pems.dot.ca.gov/'):
    """
    open a session on pems and download the required data
    :param stations: list of stations
    :param authencication_data: credential
    :param url_base: base url to log-in
    :return: 
    """
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
                    time.sleep(np.random.random_integers(ts, int(1.2 * ts)))
                except ConnectionError:
                    print('ConnectionError')
                    ts = ts * 2
                    time.sleep(ts)  # Sleep and login again
                    c.post(url_base, data=authencication_data)  # , params=p)
                    continue
                break

def download_lat_lng(stations, aut_data):
    """
    download lat and lng for a list of station 
    :param stations: dataframe of stations info
    :param aut_data: authentication data
    :return: 
    """
    add_data = {}
    for station_id, (lat, lng) in crawlLatLng(stations.index, aut_data):
        try:
            add_data[station_id] = dict(Latitude=float(lat),
                                        Longitude=float(lng))
            print("Station {} done!".format(station_id))
        except Exception as e:
            print(e)
            print("\n\n{}".format(station_id))

    down_data = pd.DataFrame.from_dict(add_data, orient="index")
    down_data.to_csv(path.join(BASE_DIR, "lat_lng.csv"))

def download_traffic(stations, auth_data, auth_base='http://pems.dot.ca.gov/', time_interval=[(1522627200, 1523059200),
                                                                                             (1523232000, 1523664000),
                                                                                             (1523836800, 1524268800),
                                                                                             (1524441600, 1524873600)]):
    base_url = "http://pems.dot.ca.gov/?report_form=1&dnode=VDS&content=loops&tab=det_timeseries&export=text&station_id={}&s_time_id={}&e_time_id={}&tod=all&tod_from=0&tod_to=0&dow_1=on&dow_2=on&dow_3=on&dow_4=on&dow_5=on&q=flow&q2=speed&gn=5min&agg=on"
    with session() as c:
        c.post(auth_base, data=auth_data)

        for i, station_id in enumerate(stations.index):
            ts = 10  # Default time to sleep
            print("Iteration: {}".format(i))
            print('initial time to sleep {}'.format(ts))
            for j, (start_time, end_time) in enumerate(time_interval):
                url = base_url.format(station_id, start_time, end_time)
                ts_small = 2    # small sleep interval
                while True:
                    try:  # Download with 10-second sleep time breaks
                        print('try to download file: {}-{}'.format(station_id, j))
                        print('time to sleep {}'.format(ts_small))
                        # Make the request and download attached file
                        r = c.get(url)
                        if r.status_code == 200:
                            with open(ensure_dir(path.join(BASE_DIR, "{}".format(station_id), "part-{}.csv".format(j))), "w") as file:
                                file.write(r.text)
                        else:
                            raise ConnectionError("Data not obtained")
                        # save file
                        time.sleep(np.random.random_integers(ts_small, int(1.2 * ts_small)))
                    except ConnectionError:
                        print('ConnectionError')
                        ts_small = ts_small * 2
                        time.sleep(ts)  # Sleep and login again
                        c.post(auth_base, data=auth_data)
                        continue
                    break
            # sleep for a longer interval
            dt = [pd.read_csv(path.join(BASE_DIR, "{}".format(station_id), "part-{}.csv".format(i)), sep="\t")
                   for i in range(len(time_interval))]
            dt = pd.concat(dt, axis=0)
            dt["5 Minutes"] = pd.to_datetime(dt["5 Minutes"], format="%m/%d/%Y %H:%M")
            dt = dt.set_index("5 Minutes")
            dt.to_csv(ensure_dir(path.join(BASE_DIR, "stations", "{}.csv".format(station_id))))
            time.sleep(np.random.random_integers(ts, int(1.2 * ts)))

def execute():
    stations = pd.read_csv(path.join(BASE_DIR, "station_comp.csv"), index_col="ID")
    auth_data = {
        'action': 'login',
        'username': UserName,
        'password': Pass
    }
    download_traffic(stations, auth_data)

if __name__ == "__main__":
    execute()






