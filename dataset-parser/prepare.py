import os
import h3
import json
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Trajectory Prediction Learning')
    parser.add_argument('--input_dir', type=str, default="/local/data1/users/anadiri/collision/", help='Path to input dataset files')
    parser.add_argument('--output_dir', type=str, default="/local/data1/users/anadiri/data", help='Path to output directory')
    args = parser.parse_args()

    datasets = [("collision_rome7", args.input_dir + "rome_res7.csv"),
                ("collision_rome8", args.input_dir + "rome_res8.csv"),
                ("collision_rome9", args.input_dir + "rome_res9.csv"),                ("collision_geolife7", args.input_dir + "geolife_output_res7.csv"),
                ("collision_geolife8", args.input_dir + "geolife_output_res8.csv"),
                ("collision_geolife9", args.input_dir + "geolife_output_res9.csv"),
                ("collision_porto7", args.input_dir + "porto_output_res7.csv"),
                ("collision_porto8", args.input_dir + "porto_output_res8.csv"),
                ("collision_porto9", args.input_dir + "porto_output_res9.csv")]


    for dataset in datasets:
        dataset_dir = os.path.join(args.output_dir, dataset[0])
        os.makedirs(dataset_dir, exist_ok=True)

        
        df = pd.read_csv(dataset[1], header=0,
                        usecols=["count", "timestamp"])
        df = df.sort_values(by=["timestamp"])["count"].to_numpy()

        def process_count(x):
            return [item[0] for item in literal_eval(x) for i in range(item[1])]
        df = [process_count(i) for i in df]
        print(df[0])
        print(len(df[0]))
        exit()

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

        df_mapped = [[str(mapping[j]) for j in i] for i in df]
        with open(os.path.join(dataset_dir, 'data.txt'), 'w') as mapping_file:
            for item in df_mapped:
                mapping_file.write(' '.join(item) + " " +
                                str(mapping['EOT']) + "\n")
