import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from ast import literal_eval
import argparse

# Parse command-line arguments for dataset directory and window size
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Geolife dataset parser.')
    parser.add_argument('--dataset_directory', type=str, default="/local/data1/shared_data/higher_order_trajectory/geolife/time_steped/",
                        help='the directory where the dataset is located')
    parser.add_argument('--window_size', type=int, default=60,
                        help='the size of the collision window')

    args = parser.parse_args()

    dataset_directory = args.dataset_directory
    window_size = args.window_size
    # Loop through the specified dataset files
    for dataset_name in ["geolife_output_res7.csv", "geolife_output_res8.csv", "geolife_output_res9.csv"]:
        time_interval = 5  # The time interval between each record in the dataset

        dataset_path = dataset_directory + dataset_name  # Construct the full dataset path

        # Load the dataset, keeping only relevant columns
        data = pd.read_csv(dataset_path)
        data = data[['user_id', 'timestamp', 'count']]

        # Normalize timestamps to start from 0
        data['timestamp'] = data['timestamp'].apply(
            lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))).astype(int)
        start_time = min(data['timestamp'])
        data['timestamp'] = data['timestamp'] - start_time
        
        # Function to process each row, expanding 'count' into time windows
        def process(row):
            time = row['timestamp']
            x = literal_eval(row['count'])
            x = [item[0] for item in x for i in range(item[1])]
            res = dict()
            for item in range(len(x)):
                window = (time+item*time_interval)//window_size
                res[window] = x[item]
            return res.items()

        # Apply processing to expand 'count' into individual records for each time window
        data['count'] = data.apply(process, axis=1)
        data.drop(['timestamp'], axis=1, inplace=True)

        # Explode the 'count' column to separate rows for each location and time window
        data = data.explode('count', ignore_index=True)

        # Extract location and time window from 'count' and drop unnecessary columns
        data['location'] = data['count'].apply(lambda x: x[1])
        data['time'] = data['count'].apply(lambda x: x[0])
        data.drop(['count'], axis=1, inplace=True)
        data['user_id'] = data['user_id'].astype(int)

        # Group by time and location to identify potential collisions
        data = data.groupby(['time', 'location']).agg(list).reset_index()

        # Filter for entries where more than one taxi is present
        data = data[data['user_id'].apply(len) > 1]

        # Generate pairs of taxi IDs for each potential collision
        data['pairs'] = data['user_id'].apply(
            lambda x: list(itertools.combinations(x, 2)))
        data.drop('user_id', axis=1, inplace=True)
        data = data.explode('pairs')
        data['taxi1'] = data['pairs'].apply(lambda x: x[0])
        data['taxi2'] = data['pairs'].apply(lambda x: x[1])
        data.drop('pairs', axis=1, inplace=True)

        # Finalize the dataset and write to a new CSV file
        data = data[['taxi1', 'taxi2', 'time', 'location']]
        data.to_csv("/local/data1/users/anadiri/collision/" + "collision_" +
                    str(window_size) + "_" + dataset_name, index=False)