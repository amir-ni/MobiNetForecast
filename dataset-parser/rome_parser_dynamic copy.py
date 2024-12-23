import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from ast import literal_eval
import argparse

# Parse command-line arguments for dataset directory and window size
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rome dataset parser.')
    parser.add_argument('--dataset_directory', type=str, default="/local/data1/users/anadiri/collision/",
                        help='the directory where the dataset is located')
    parser.add_argument('--window_size', type=int, default=300,
                        help='the size of the collision window')

    args = parser.parse_args()

    dataset_directory = args.dataset_directory
    # window_size = args.window_size
    for window_size in [7,15,30,60,90,120,180,300]:
        # Loop through the specified dataset files
        for dataset_name in ["rome_res7.csv", "rome_res8.csv", "rome_res9.csv"]:
            time_interval = 7  # The time interval between each record in the dataset

            dataset_path = dataset_directory + dataset_name  # Construct the full dataset path

            # Load the dataset, keeping only relevant columns
            data = pd.read_csv(dataset_path)
            data = data[['taxi_id', 'timestamp', 'count']]

            # Normalize timestamps to start from 0
            data['timestamp'] = data['timestamp'].apply(
                lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f%z")))
            start_time = min(data['timestamp'])
            data['timestamp'] = data['timestamp'] - start_time
            
            # Function to process each row, expanding 'count' into time windows
            def process(row):
                time = row['timestamp']
                x = literal_eval(row['count'])
                x = [item[0] for item in x for i in range(item[1])]
                res = dict()
                for item in range(len(x)):
                    window = int((time+item*time_interval)/window_size)
                    res[window] = (x[item], window)
                return list(res.values())

            # Apply processing to expand 'count' into individual records for each time window
            data['count'] = data.apply(process, axis=1)

            # Explode the 'count' column to separate rows for each location and time window
            data = data.explode('count', ignore_index=True)

            # Extract location and time window from 'count' and drop unnecessary columns
            data['location'] = data['count'].apply(lambda x: x[0])
            data['time'] = data['count'].apply(lambda x: x[1])
            data.drop(['count', 'timestamp'], axis=1, inplace=True)
            data['taxi_id'] = data['taxi_id'].astype(int)

            # Group by time and location to identify potential collisions
            data = data.groupby(['time', 'location']).agg(list).reset_index()

            # Filter for entries where more than one taxi is present
            data = data[data['taxi_id'].apply(len) > 1]

            # Generate pairs of taxi IDs for each potential collision
            

            # Generate pairs of taxi IDs for each potential collision
            data['pairs'] = data['taxi_id'].apply(
                lambda x: [(a[1],a[0]) if a[0] > a[1] else (a[0],a[1]) for a in itertools.combinations(x, 2)])
            data = data.drop('taxi_id', axis=1)
            data = data.explode('pairs')
            print(dataset_name, window_size, data['pairs'].nunique(), data.shape[0])
            continue
            data[['taxi1', 'taxi2']] = pd.DataFrame(
                data['pairs'].tolist(), index=data.index)
            data = data.drop('pairs', axis=1)
            
            # Finalize the dataset and write to a new CSV file
            data.to_csv(dataset_directory + "collision_" + str(window_size) +
                        "_" + dataset_name, index=False)
