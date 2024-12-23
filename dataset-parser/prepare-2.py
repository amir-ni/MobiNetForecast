import os
import h3
import json
import numpy as np
import pandas as pd
from datetime import datetime
from ast import literal_eval
import argparse

# Parse command-line arguments for dataset directory and window size
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rome dataset parser.')
    parser.add_argument('--dataset_directory', type=str, default="/local/data1/users/anadiri/collision/",
                        help='the directory where the dataset is located')
    parser.add_argument('--window_size', type=int, default=60,
                        help='the size of the collision window')
    parser.add_argument('--output_dir', type=str, default="/local/data1/users/anadiri/data", help='Path to output directory')
    parser.add_argument('--time_interval', type=int, default=7)
    
    args = parser.parse_args()

    datasets = [("collision_rome7", args.dataset_directory + "rome_res7.csv"),
                ("collision_rome8", args.dataset_directory + "rome_res8.csv"),
                ("collision_rome9", args.dataset_directory + "rome_res9.csv")]
    
    dataset_directory = args.dataset_directory
    window_size = args.window_size
    time_interval = args.time_interval
    # Loop through the specified dataset files
    for dataset in datasets:
        dataset_dir = os.path.join(args.output_dir, dataset[0])
        os.makedirs(dataset_dir, exist_ok=True)

        # Load the dataset, keeping only relevant columns
        data = pd.read_csv(dataset[1])
        data = data[['timestamp', 'count']]

        # Normalize timestamps to start from 0
        data['timestamp'] = data['timestamp'].apply(
            lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f%z"))).astype(int)
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
            return list(res.values())

        # Apply processing to expand 'count' into individual records for each time window
        data['count'] = data.apply(process, axis=1)
        data['timestamp'] = data['timestamp']//window_size

        
        # windows_size = max([list(i.keys())[-1] for i in data["count"]])
        # train_windows_size = int(windows_size * 0.8)

        # def process2(row):
        #     return {k: v for k, v in row['count'].items() if k <= train_windows_size}

        # def process3(row):
        #     return {k: v for k, v in row['count'].items() if k > train_windows_size}

        # data['train'] = data.apply(process2, axis=1)
        # data['test'] = data.apply(process3, axis=1)

        df = data["count"].to_list()
        vocab = ["EOT"] + list(np.unique(np.concatenate(df, axis=0)))
        with open(os.path.join(dataset_dir, 'vocab.txt'), 'w') as vocab_file:
            for item in vocab:
                vocab_file.write(item+"\n")

        mapping = {k: v for v, k in enumerate(vocab)}
        with open(os.path.join(dataset_dir, 'mapping.json'), 'w') as mapping_file:
            mapping_file.write(json.dumps(mapping))

        neighbors = dict()
        for x in vocab[1:]:
            neighbors[mapping[str(x)]] = [mapping[i]
                                        for i in h3.hex_ring(str(x)) if i in vocab]
        with open(os.path.join(dataset_dir, 'neighbors.json'), 'w') as neighbors_file:
            neighbors_file.write(json.dumps(neighbors))

        with open(os.path.join(dataset_dir, 'data.txt'), 'w') as mapping_file:
            def process2(row):
                a = [str(mapping[i]) for i in row['count']]
                a = str(row['timestamp']) + " " + ' '.join(a) + " " + str(mapping['EOT']) + "\n"
                mapping_file.write(a)
            data.apply(process2, axis=1)
