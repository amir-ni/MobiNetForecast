import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from ast import literal_eval

for dataset_name in ["rome_res7.csv"]:#,"rome_res8.csv","rome_res9.csv"]:
    time_interval = 7
    dataset_directory = "/local/data1/users/anadiri/collision/"

    dataset_path = dataset_directory + dataset_name

    data = pd.read_csv(dataset_path)
    data = data[['taxi_id','timestamp','count']]

    data['timestamp'] = data['timestamp'].apply(lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f%z")))
    start_time = min(data['timestamp'])
    data['timestamp'] = data['timestamp'] - start_time
    data['timestamp'] = data['timestamp']//time_interval

    def process(x):
        x = literal_eval(x)
        x = [item[0] for item in x for i in range(item[1])]
        x = [(x[item],item) for item in range(len(x))]
        return x

    data['count'] = data['count'].apply(process)

    data = data.explode('count', ignore_index=True)

    data['location'] = data['count'].apply(lambda x: x[0])
    data['time_passed'] = data['count'].apply(lambda x: x[1])

    data['time'] = (data['time_passed'] + data['timestamp']).astype(int)

    data['taxi_id'] = data['taxi_id'].astype(int)

    data.drop(['count','time_passed','timestamp'], axis=1, inplace=True)

    data = data.groupby(['time','location']).agg(list).reset_index()

    data = data[data['taxi_id'].apply(len) > 1]

    data['pairs'] = data['taxi_id'].apply(lambda x: list(itertools.combinations(x, 2)))
    data = data.drop('taxi_id', axis=1)
    data = data.explode('pairs')
    data[['taxi1', 'taxi2']] = pd.DataFrame(data['pairs'].tolist(), index=data.index)
    data = data.drop('pairs', axis=1)

    data.to_csv(dataset_directory + "collision_" + dataset_name, index=False)
