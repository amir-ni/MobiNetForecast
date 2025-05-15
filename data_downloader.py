import argparse
from typing import List, Dict, Optional, Union, Any
import csv
import itertools
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
from pyproj import Transformer
import numpy as np
import pandas as pd
import requests
import h3
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ------------------------------------------------------
# Constants
# ------------------------------------------------------

GEOLIFE_URL = (
    "https://download.microsoft.com/download/F/4/8/"
    "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/"
    "Geolife%20Trajectories%201.3.zip"
)
SFCO_AA_URL = "https://osf.io/download/za5kc/"
SFCO_AB_URL = "https://osf.io/download/k8y49/"
DEFAULT_RESOLUTIONS = [8, 9 , 10]
WINDOW_SIZE = 300 # seconds
SECONDS_PER_DAY = 24 * 60 * 60
# Global split: all trajectories with start < this time are training.
GEOLIFE_TRAIN_SPLIT = "20110401000000" # Format: YYYYMMDDHHMMSS
SFCO_TRAIN_SPLIT = "20190724000000" # Format: YYYYMMDDHHMMSS

DAILY_SPLIT_HOUR = 13 # 1 PM

# Minimum number of time steps required in the input for test trajectories.
MIN_INFERENCE_ACTIVITY = 12

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------

def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024, timeout: int = 30) -> None:
    """
    Downloads a file from the given URL to the specified destination path.

    Parameters:
        url (str): URL of the file to download.
        dest_path (Path): Destination file path.
        chunk_size (int, optional): Size (in bytes) of each chunk. Defaults to 1MB.
        timeout (int, optional): Request timeout in seconds. Defaults to 30 seconds.

    Returns:
        None
    """
    print(f"Downloading file from {url} to {dest_path}...")
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as file, tqdm(
        desc=f"Downloading {dest_path.name}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    print("Download complete.")


def unzip_file(zip_path: Path, extract_to: Path) -> None:
    """
    Unzips the given zip file into the specified directory.

    Parameters:
        zip_path (Path): Path to the zip file.
        extract_to (Path): Directory where the contents will be extracted.

    Returns:
        None
    """
    print(f"Extracting {zip_path} to {extract_to}...")
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def parse_geolife_data(data_dir: Union[str, Path], output_csv: Union[str, Path]) -> None:
    """
    Parses GeoLife .plt files and writes a CSV with the following columns:
      UserID, TrajectoryID, Latitude, Longitude, AbsoluteTimestamp.

    Parameters:
        data_dir (Union[str, Path]): Directory containing the GeoLife data.
        output_csv (Union[str, Path]): File path for the output CSV.

    Returns:
        None
    """
    print("Parsing GeoLife trajectory data...")
    data_dir = Path(data_dir) / "Geolife Trajectories 1.3" / "Data"
    all_entries: List[List[Any]] = []
    tz = timezone(timedelta(hours=-8))

    for user_dir in tqdm(list(data_dir.iterdir()), desc="Processing Users"):
        if not user_dir.is_dir():
            continue

        trajectory_dir = user_dir / "Trajectory"
        if not trajectory_dir.exists():
            continue

        for traj_file in trajectory_dir.glob("*.plt"):
            if traj_file.name == "20000101231219.plt":
                print(
                    "Skipped an invalid file (timestamp 2000/01/01 - 23:12:19 "
                    "is not in range of the GeoLife dataset data gathering period)."
                )
                continue

            with traj_file.open("r", encoding="utf-8") as file:
                lines = file.readlines()

            data_lines = lines[6:]
            for line in data_lines:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue

                latitude, longitude = float(parts[0]), float(parts[1])
                date_string = f"{parts[5]} {parts[6]}"
                abs_timestamp = int(datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")\
                    .replace(tzinfo=tz).timestamp())
                all_entries.append([user_dir.name, traj_file.stem,
                    latitude, longitude, abs_timestamp])

    if not all_entries:
        print("No entries found.")
        return

    # Sort by AbsoluteTimestamp
    all_entries.sort(key=lambda x: x[4])

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["UserID", "TrajectoryID",
            "Latitude", "Longitude", "AbsoluteTimestamp"])
        csv_writer.writerows(all_entries)

    print(f"Parsed data saved to {output_csv}.")


def gps_to_h3(latitude: float, longitude: float, resolution: int) -> str:
    """
    Converts GPS coordinates to an H3 hexagonal cell.

    Parameters:
        latitude (float): Latitude coordinate.
        longitude (float): Longitude coordinate.
        resolution (int): H3 resolution level.

    Returns:
        str: The H3 cell identifier.
    """
    return h3.latlng_to_cell(latitude, longitude, resolution)


def add_h3_columns(df: pd.DataFrame, resolutions: List[int]) -> pd.DataFrame:
    """
    Adds H3 cell columns to the DataFrame for each given resolution.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Latitude' and 'Longitude' columns.
        resolutions (List[int]): List of H3 resolution levels to compute.

    Returns:
        pd.DataFrame: DataFrame with additional columns for each H3 resolution.
    """
    for res in resolutions:
        col = f"resolution{res}"
        df[col] = df.apply(
            lambda row, res=res: gps_to_h3(row["Latitude"], row["Longitude"], res),
            axis=1
        )
    return df


def remove_consecutive_duplicates(seq: List[Any]) -> List[Any]:
    """
    Removes consecutive duplicate elements from a list.

    Parameters:
        seq (List[Any]): Input list of elements.

    Returns:
        List[Any]: List with consecutive duplicates removed.
    """
    return [x for i, x in enumerate(seq) if i == 0 or x != seq[i-1]]


# ------------------------------------------------------
# Preprocessing: Grouping & Splitting Trajectories
# ------------------------------------------------------

def preprocess_trajectories(
    output_csv: Union[str, Path],
    resolutions: List[int],
    window_size: int,
    data_dir: Union[str, Path],
    global_train_split_str: str,
    daily_split_hour: int,
    min_inference_activity: int
) -> None:
    """
    Processes the parsed CSV file to create training and testing sequences.

    For trajectories starting before the global training split, the full sequence is used.
    For trajectories starting after the split, the sequence is split at the daily split hour:
      - Points before the split form the input for prediction.
      - Points after the split are used for evaluation.
      - Only trajectories with at least the minimum inference activity are kept.

    Saves separate CSV files for training and testing sequences for each resolution.

    Parameters:
        output_csv (Union[str, Path]): File path of the parsed CSV.
        resolutions (List[int]): List of H3 resolution levels.
        window_size (int): Time window size in seconds.
        data_dir (Union[str, Path]): Base directory for storing processed data.
        global_train_split_str (str): Global split time in the format YYYYMMDDHHMMSS.
        daily_split_hour (int): Hour of the day to split test trajectories.
        min_inference_activity (int): Minimum number of time windows required for test trajectories.

    Returns:
        None
    """
    df = pd.read_csv(output_csv)
    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)
    df["AbsoluteTimestamp"] = df["AbsoluteTimestamp"].astype(int)

    # Add H3 columns for each resolution
    df = add_h3_columns(df, resolutions)

    # Create output directory for GeoLife data.
    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Convert the global training split to a timestamp.
    global_train_split_ts = int(datetime.strptime(global_train_split_str, "%Y%m%d%H%M%S")\
        .replace(tzinfo=timezone.utc).timestamp())

    for res in resolutions:
        col = f"resolution{res}"
        df_res = df[["UserID", "TrajectoryID", col, "AbsoluteTimestamp"]].copy()
        grouped = df_res.groupby(["UserID", "TrajectoryID"]).agg({
            col: list,
            "AbsoluteTimestamp": "min"
        }).reset_index()
        grouped["HexagonSequence"] = grouped[col].apply(remove_consecutive_duplicates)

        train_rows = []
        test_rows = []

        for _, row in grouped.iterrows():
            user = row["UserID"]
            traj = row["TrajectoryID"]
            seq = row["HexagonSequence"]
            start_time = row["AbsoluteTimestamp"]

            if start_time < global_train_split_ts:
                if len(seq) < min_inference_activity:
                    continue
                train_rows.append([user, traj, " ".join(map(str, seq)), start_time])
            else:
                day_start = start_time // SECONDS_PER_DAY * SECONDS_PER_DAY
                split_ts = day_start + daily_split_hour * 60 * 60

                split_index = len(row[col])
                for i in range(len(row[col])):
                    point_ts = start_time + i * window_size
                    if point_ts >= split_ts:
                        split_index = i
                        break
                input_seq = remove_consecutive_duplicates(row[col][:split_index])
                pred_seq = remove_consecutive_duplicates(row[col][split_index:])
                if len(input_seq) < min_inference_activity or len(pred_seq) == 0:
                    continue
                if pred_seq[0] == input_seq[-1]:
                    if len(pred_seq) < 2:
                        continue
                    pred_seq = pred_seq[1:]

                test_rows.append([user, traj,
                                  " ".join(map(str, input_seq[-min_inference_activity:])),
                                  " ".join(map(str, pred_seq[-min_inference_activity:])),
                                  split_ts - (min_inference_activity * window_size)])
        res_dir = base_dir / f"resolution-{res}"
        res_dir.mkdir(parents=True, exist_ok=True)
        train_file = res_dir / "train_sequences.csv"
        test_file = res_dir / "test_sequences.csv"

        pd.DataFrame(train_rows,\
          columns=["UserID", "TrajectoryID", "HexagonSequence", "StartTime"])\
          .to_csv(train_file, index=False)
        print(f"Training sequences for resolution {res} saved in {train_file}.")

        pd.DataFrame(test_rows, columns=["UserID", "TrajectoryID",\
          "InputSequence", "PredictionSequence", "StartTime"])\
          .to_csv(test_file, index=False)
        print(f"Testing sequences for resolution {res} saved in {test_file}.")



# ------------------------------------------------------
# Embeddings, Vocab & Mapping
# ------------------------------------------------------

def generate_embeddings(
    vocab: List[str],
    embedding_dim: int,
    mean: float = 0,
    std: float = 0.02,
    projection_matrix: Optional[np.ndarray] = None,
    random_seed: int = 10
) -> np.ndarray:
    """
    Generates embeddings for a vocabulary based on axial coordinates.
    Assumes the first token is 'EOT'.

    Parameters:
        vocab (List[str]): List of tokens in the vocabulary.
        embedding_dim (int): Dimension of the embedding vectors.
        mean (float, optional): Mean for the normal distribution. Defaults to 0.
        std (float, optional): Standard deviation for the normal distribution. Defaults to 0.02.
        projection_matrix (Optional[np.ndarray], optional): Projection matrix.
        random_seed (int, optional): Seed for random number generation. Defaults to 10.

    Returns:
        np.ndarray: Generated embeddings as a NumPy array.
    """
    origin_hex = vocab[1] if len(vocab) > 1 else vocab[0]
    base_i, base_j = h3.cell_to_local_ij(origin_hex, origin_hex)
    axial_coordinates = []
    for h3_hex in vocab[1:]:
        target_i, target_j = h3.cell_to_local_ij(origin_hex, h3_hex)
        axial_coordinates.append((target_i - base_i, target_j - base_j))
    np.random.seed(random_seed)
    if projection_matrix is None:
        projection_matrix = np.random.randn(2, embedding_dim)
    projected = np.dot(axial_coordinates, projection_matrix)
    eot_embedding = np.random.normal(loc=mean, scale=std, size=(1, embedding_dim))
    normal_samples = np.random.normal(loc=mean, scale=std, size=(len(vocab) - 1) * embedding_dim)
    flat_proj = projected.flatten()
    sorted_indices = np.argsort(flat_proj)
    sorted_proj = np.empty_like(flat_proj)
    sorted_proj[sorted_indices] = np.sort(normal_samples)
    sorted_proj = sorted_proj.reshape(projected.shape)
    embeddings = np.concatenate((eot_embedding, sorted_proj), axis=0)
    return embeddings


def process_embeddings_and_vocab(
    data_dir: Union[str, Path],
    resolutions: List[int],
    embedding_dim: Optional[int],
    min_inference_activity: int
) -> None:
    """
    Processes training sequences to create vocabulary, mapping,
    neighbor relations, and embeddings (if specified).

    For each resolution, the following files are generated:
      - vocab.txt (with 'EOT' as the first token)
      - mapping.json (mapping from hex cell to index)
      - neighbors.json (neighbor relationships for each hex cell)
      - embeddings.npy (if embedding_dim is provided)
      - data_train.csv and data_test.csv (indexed sequences for training and testing)

    Parameters:
        data_dir (Union[str, Path]): Base directory where processed data is stored.
        resolutions (List[int]): List of H3 resolution levels.
        embedding_dim (Optional[int]): Dimension of the embedding vectors.
                                       If None, embeddings are not generated.
        min_inference_activity (int): Minimum number of windows (time steps) required

    Returns:
        None
    """
    base_dir = Path(data_dir)
    for res in resolutions:
        res_dir = base_dir / f"resolution-{res}"
        train_file = res_dir / "train_sequences.csv"
        test_file = res_dir / "test_sequences.csv"
        if not train_file.exists():
            print(f"Warning: {train_file} does not exist. Skipping resolution {res}.")
            continue

        df_train = pd.read_csv(train_file)
        sequences = df_train["HexagonSequence"].tolist()
        if test_file.exists():
            df_test = pd.read_csv(test_file)
            sequences += df_test["InputSequence"].tolist() + df_test["PredictionSequence"].tolist()
        split_seqs = [seq.split() for seq in sequences]
        vocab = list(np.unique(np.concatenate(split_seqs, axis=0)))
        vocab = ["EOT"] + vocab # Prepend 'EOT' as the first token

        vocab_file = res_dir / "vocab.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab) + "\n")
        print(f"Vocab for resolution {res} saved in {vocab_file}.")

        mapping = {token: idx for idx, token in enumerate(vocab)}
        mapping_file = res_dir / "mapping.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)

        neighbors: Dict[int, List[int]] = {}
        for token in vocab[1:]:
            ring = h3.grid_ring(token)
            neighbors[mapping[token]] = [mapping[n] for n in ring if n in mapping]
        neighbors_file = res_dir / "neighbors.json"
        with open(neighbors_file, "w", encoding="utf-8") as f:
            json.dump(neighbors, f, ensure_ascii=False)

        if embedding_dim is not None:
            embeddings = generate_embeddings(vocab, embedding_dim)
            embeddings_file = res_dir / "embeddings.npy"
            np.save(embeddings_file, embeddings)
            print(f"Embeddings for resolution {res} saved in {embeddings_file}.")

        df_train = df_train[
            df_train["HexagonSequence"]
            .str.split()
            .str.len()
            .ge(min_inference_activity)
        ]
        df_train["HexagonSequence"] = df_train["HexagonSequence"].apply(
            lambda seq, m=mapping: " ".join(str(m[t]) for t in seq.split()) + f" {m['EOT']}"
        )
        train_index_file = res_dir / "data_train.csv"
        df_train.to_csv(train_index_file, index=False)
        print(f"Processed training data for resolution {res} saved in {train_index_file}.")

        if test_file.exists():
            df_test = pd.read_csv(test_file)
            df_test = df_test[
                df_test["InputSequence"]
                .str.split()
                .str.len()
                .ge(min_inference_activity)
            ]
            df_test["InputSequence"] = df_test["InputSequence"].apply(
                lambda seq, m=mapping: " ".join(str(m[t]) for t in seq.split())
            )
            df_test["PredictionSequence"] = df_test["PredictionSequence"].apply(
                lambda seq, m=mapping: " ".join(str(m[t]) for t in seq.split())
            )
            test_index_file = res_dir / "data_test.csv"
            df_test.to_csv(test_index_file, index=False)
            print(f"Processed testing data for resolution {res} saved in {test_index_file}.")


# ------------------------------------------------------
# Edges (Collision Graph) Generation
# ------------------------------------------------------

def generate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates pairs of users (edges) that share the same time window and location.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['UserID', 'time', 'location', 'seen'].

    Returns:
        pd.DataFrame: DataFrame with collision edges (columns: user1, user2, time, location).
    """
    cols = ["time", "location"] + (["seen"] if "seen" in df.columns else [])
    grouped = df.groupby(cols)["UserID"].apply(list).reset_index()
    return pd.DataFrame([
        dict(
            user1=u,
            user2=v,
            **{k: row[k] for k in cols}
        )
        for row in grouped.to_dict("records")
        for u, v in itertools.combinations(sorted(set(row["UserID"])), 2)
    ], columns=["user1", "user2"] + cols)


def generate_edges_for_resolution(res_dir: Path, window_size: int, daily_split_hour: int) -> None:
    """
    Generates collision edges for a given resolution directory.

    - For training edges, all points from trajectories in train_sequences.csv and
      points from the InputSequence of test trajectories (i.e., before the daily split) are used.
    - For test edges, points from the PredictionSequence of test trajectories are used.

    Each point's time window is computed as:
        time_window = (point_time // window_size) + 1

    Parameters:
        res_dir (Path): Directory corresponding to a specific resolution.
        window_size (int): Time window size in seconds.
        daily_split_hour (int): Hour of the day (0-23) to split test trajectories.

    Returns:
        None
    """
    train_points: List[tuple] = [] # List of tuples: (UserID, time, location)
    test_points: List[tuple] = [] # List of tuples: (UserID, time, location, seen)

    train_file = res_dir / "train_sequences.csv"
    test_file = res_dir / "test_sequences.csv"

    # Process training trajectories.
    if train_file.exists():
        df_train = pd.read_csv(train_file)
        for _, row in df_train.iterrows():
            user = row["UserID"]
            start_time = int(row["StartTime"])
            seq = row["HexagonSequence"].split()
            for i, cell in enumerate(seq):
                point_ts = start_time + i * window_size
                time_window = (point_ts // window_size) + 1
                train_points.append((user, time_window, cell))

    # Process test trajectories.
    if test_file.exists():
        df_test = pd.read_csv(test_file)
        for _, row in df_test.iterrows():
            user = row["UserID"]
            start_time = int(row["StartTime"])
            # Compute the daily split timestamp.
            day_start = start_time // SECONDS_PER_DAY * SECONDS_PER_DAY
            daily_split_ts = day_start + daily_split_hour * 60 * 60

            # Process InputSequence (for training edges: before daily split).
            input_seq = row["InputSequence"].split()
            for i, cell in enumerate(input_seq):
                point_ts = start_time + i * window_size
                if point_ts < daily_split_ts:
                    time_window = (point_ts // window_size) + 1
                    test_points.append((user, time_window, cell, True))
            input_length = len(input_seq)
            pred_seq = row["PredictionSequence"].split()
            for j, cell in enumerate(pred_seq):
                point_ts = start_time + (input_length + j) * window_size
                time_window = (point_ts // window_size) + 1
                test_points.append((user, time_window, cell, False))

    # Create DataFrames from the collected points.
    train_df = pd.DataFrame(train_points, columns=["UserID", "time", "location"])\
        .drop_duplicates(subset=['UserID', 'time'], keep='last')
    test_df = pd.DataFrame(test_points, columns=["UserID", "time", "location", "seen"])\
        .drop_duplicates(subset=['UserID', 'time'], keep='last')
    train_edges_df = generate_pairs(train_df)
    test_edges_df = generate_pairs(test_df)

    train_edges_file = res_dir / "edges_train.csv"
    test_edges_file = res_dir / "edges_test.csv"
    train_edges_df.to_csv(train_edges_file, index=False)
    test_edges_df.to_csv(test_edges_file, index=False)
    all_nodes = set(train_edges_df["user1"]).union(train_edges_df["user2"]) \
        .union(test_edges_df["user1"]).union(test_edges_df["user2"])
    node2idx = {user: idx for idx, user in enumerate(sorted(all_nodes))}
    mapping_file = res_dir / "node2idx.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(node2idx, f, ensure_ascii=False)
    print(f"Edges for {res_dir.name} saved to {res_dir}")


def generate_edges(
    data_dir: Union[str, Path],
    resolutions: List[int],
    window_size: int,
    daily_split_hour: int
) -> None:
    """
    Generates collision edges for all specified resolutions.

    Iterates over each resolution directory to generate collision (edge) files.

    Parameters:
        data_dir (Union[str, Path]): Base directory where processed data is stored.
        resolutions (List[int]): List of H3 resolution levels.
        window_size (int): Time window size in seconds.
        daily_split_hour (int): Hour of the day (0-23) to split test trajectories.

    Returns:
        None
    """
    base_dir = Path(data_dir)
    for res in resolutions:
        res_dir = base_dir / f"resolution-{res}"
        generate_edges_for_resolution(res_dir, window_size, daily_split_hour)


# ------------------------------------------------------
# Cleanup Function (Optional)
# ------------------------------------------------------

def cleanup(*items: Path) -> None:
    """
    Recursively deletes the specified files and directories.

    Parameters:
        *items (Path): Files or directories to remove.

    Returns:
        None
    """
    for item in items:
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            for sub in item.iterdir():
                cleanup(sub)
            item.rmdir()


def convert_sfco(
    input_tsv: Path,
    output_csv: Path,
    days_to_keep: int
) -> None:
    """
    Converts the SFCO dataset from TSV to CSV format.
    Extracts latitude and longitude from the 'location' column.
    Converts them to WGS84 coordinates.
    Converts the 'simulationTime' column to a UNIX timestamp.
    Filters the data to keep only the specified number of days.
    Saves the processed data to a CSV file.
    Parameters:
        input_tsv (Path): Path to the input TSV file.
        output_csv (Path): Path to the output CSV file.
        days_to_keep (int): Number of days to keep in the dataset.
    Returns:
        None
    """
    num_users = 3000
    sampling_freq = 300
    read_rows = num_users * days_to_keep * 24 * (60 * 60 // sampling_freq)
    df = pd.read_csv(input_tsv, sep='\t', nrows=read_rows)\
        .rename(columns={'simulationTime': 'AbsoluteTimestamp', 'agentId': 'UserID'})

    df[['Latitude', 'Longitude']] = df['location'].str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)')
    df.drop(columns=['location'], inplace=True)

    df['Longitude'] = pd.to_numeric(df['Longitude'])
    df['Latitude'] = pd.to_numeric(df['Latitude'])

    transformer = Transformer.from_crs("epsg:26910", "epsg:4326")
    df['Latitude'], df['Longitude'] = \
        transformer.transform(df['Latitude'].values, df['Longitude'].values)

    df['AbsoluteTimestamp'] = pd.to_datetime(df['AbsoluteTimestamp'], format="%Y-%m-%dT%H:%M:%S.%f")

    df['AbsoluteTimestamp'] = df['AbsoluteTimestamp'].view('int64') // 10**9

    df['TrajectoryID'] = df['AbsoluteTimestamp'] // (24*60*60)

    df.to_csv(output_csv, index=False)


def generate_stats(
    data_dir: Union[str, Path],
    resolutions: List[int],
    window_size: int
) -> None:
    """
    Generates statistics about the dataset.
    Returns:
        None
    """
    base_dir = Path(data_dir)
    for res in resolutions:
        res_dir = base_dir / f"resolution-{res}"
        train_df = pd.read_csv(res_dir / "edges_train.csv",
                            usecols=["user1","user2","time"])
        test_df = pd.read_csv(res_dir / "edges_test.csv",
                            usecols=["user1","user2","time","seen"])
        if len(test_df) == 0 or len(train_df) == 0:
            print(f"Warning: {res_dir} has no edges.")
            continue
        train_df['seen'] = True
        test_df["seen"]  = test_df["seen"].astype(bool)
        df = pd.concat([train_df, test_df],
                      ignore_index=True,
                      sort=False)
        df['time'] = (df['time'].astype(int) * window_size) // (24 * 60 * 60)
        df['edge'] = list(zip(df['user1'], df['user2']))
        flag_sets = df.groupby('edge')['seen'].agg(set)
        both_seen_and_unseen = flag_sets[flag_sets == {True, False}].index
        intersection_size = len(both_seen_and_unseen)
        seen_edges = set(df.loc[df['seen'], 'edge'])
        unseen_edges = set(df.loc[~df['seen'], 'edge'])
        reoccurrence = intersection_size / len(seen_edges) if len(seen_edges) > 0 else 0
        surprise = len(unseen_edges - seen_edges) / len(unseen_edges) \
            if len(unseen_edges) > 0 else 0

        seen_so_far = set()
        tea_rows = []
        novelty_vals = []
        min_t = sorted(df["time"].unique())[0]
        max_t = sorted(df["time"].unique())[-1]

        for t in range(min_t, max_t + 1):
            Et = set(df.loc[df["time"] == t, "edge"])
            new_edges      = Et - seen_so_far
            repeated_edges = Et & seen_so_far

            tea_rows.append({"time": t-min_t,
                             "New": len(new_edges),
                             "Repeated": len(repeated_edges)})

            novelty_vals.append(len(new_edges) / len(Et) if len(Et) > 0 else 0)
            seen_so_far |= Et

        novelty_index = sum(novelty_vals) / len(novelty_vals)

        tea_df = pd.DataFrame(tea_rows).set_index("time")
        fig, ax = plt.subplots()
        colors = ['#B3B3B3', '#CD2E46']
        tea_df[["Repeated", "New"]].plot(
            kind="bar",
            stacked=True,
            color=colors,
            ax=ax,
        )
        for bar in ax.containers[1]:
            bar.set_hatch('//')

        split_time = min(set(df.loc[~df['seen'], 'time'])) - min_t
        ax.axvline(x=split_time, color='blue', linestyle='--', linewidth=2)
        ax.plot([split_time], [0],
                marker='*',
                markersize=15,
                color='blue',
                clip_on=False)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.legend(loc='upper left',
                        frameon=True,
                        framealpha=0.3,
                        facecolor='white')
        ax.set_xlabel("Day")
        ax.set_ylabel("Number of edges")
        fig.savefig(res_dir / "tea_plot.png", dpi=300)
        plt.tight_layout()
        fig.savefig(res_dir / "tea_plot.pdf")
        plt.close(fig)

        with open(res_dir / 'stats.txt', 'w', encoding="utf-8") as stats_file:
            stats_file.write(f"Resolution {res}:\n")
            stats_file.write(f"  - # Total edges:      {df['edge'].nunique()}\n")
            stats_file.write(f"  - # Training edges:   {len(seen_edges)}\n")
            stats_file.write(f"  - # Evaluation edges: {len(unseen_edges)}\n")
            stats_file.write(f"  - # Repeated edges:   {intersection_size}\n")
            stats_file.write(f"  - Reoccurrence:       {reoccurrence:.4f}\n")
            stats_file.write(f"  - Surprise:           {surprise:.4f}\n")
            stats_file.write(f"  - Novelty Index:      {novelty_index:.4f}\n\n")
    return


def geolife(cfg: argparse.Namespace) -> None:
    """
    Main function that orchestrates the trajectory processing pipeline.

    Parses command-line arguments and executes the following steps:
      - Downloading and unzipping the GeoLife dataset.
      - Parsing the dataset.
      - Preprocessing trajectories.
      - Processing embeddings and vocabulary.
      - Generating collision edges.
      - Optionally cleaning up intermediate files.

    Returns:
        None
    """
    print("Processing GeoLife dataset...")
    data_dir = Path(cfg.data_dir)
    geolife_dir = data_dir / "geolife"
    geolife_dir.mkdir(parents=True, exist_ok=True)

    zip_path = geolife_dir / "GeoLife_Trajectories.zip"
    extract_dir = geolife_dir / "GeoLife_Trajectories"
    output_csv = geolife_dir / "geolife_trajectories.csv"

    if not output_csv.exists():
        if not extract_dir.exists():
            if not zip_path.exists():
                download_file(GEOLIFE_URL, zip_path)
            unzip_file(zip_path, extract_dir)
            print("Zip file extracted.")
        parse_geolife_data(extract_dir, output_csv)
        print("Data converted to CSV format.")

    print("Preprocessing trajectories...")
    preprocess_trajectories(
        output_csv=output_csv,
        resolutions=cfg.resolutions,
        window_size=cfg.window_size,
        data_dir=geolife_dir,
        global_train_split_str=GEOLIFE_TRAIN_SPLIT,
        daily_split_hour=cfg.daily_split_hour,
        min_inference_activity=cfg.min_inference_activity
    )
    print("Preprocessing complete.")

    print("Processing embeddings and vocab...")
    process_embeddings_and_vocab(
        data_dir=geolife_dir,
        resolutions=cfg.resolutions,
        embedding_dim=cfg.embedding_dim,
        min_inference_activity=cfg.min_inference_activity
    )
    print("Embeddings and vocab processing complete.")

    print("Generating edges...")
    generate_edges(geolife_dir, cfg.resolutions, cfg.window_size, cfg.daily_split_hour)
    print("Edges generation complete.")
    print(generate_stats(geolife_dir, cfg.resolutions, cfg.window_size))
    print("Statistics saved to data directory.")

    if cfg.clean:
        print("Cleaning up intermediate files...")
        cleanup(zip_path, extract_dir, output_csv)
        print("Cleanup complete.")


def sfco(cfg: argparse.Namespace) -> None:
    """
    Main function that orchestrates the trajectory processing pipeline.

    Parses command-line arguments and executes the following steps:
      - Downloading and unzipping the SFCO dataset.
      - Parsing the dataset.
      - Preprocessing trajectories.
      - Processing embeddings and vocabulary.
      - Generating collision edges.
      - Optionally cleaning up intermediate files.

    Returns:
        None
    """
    print("Processing SFCO dataset...")
    data_dir = Path(cfg.data_dir)
    sfco_dir = data_dir / "sfco"
    sfco_dir.mkdir(parents=True, exist_ok=True)

    zip_AA_path = sfco_dir / "traj.tsv.zip.part-aa"
    zip_BB_path = sfco_dir / "traj.tsv.zip.part-ab"
    zip_path = sfco_dir / "traj.tsv.zip"
    extract_dir = sfco_dir / "sfco_Trajectories"
    extracted_file = (
        extract_dir
        / "home/datapaper/15months/3k/sfco/pol/examples/logs/data_wrangling"
        / "data_set/traj.tsv"
    )
    output_tsv = sfco_dir / "SFCO-3k-15mo.tsv"
    output_csv = sfco_dir / "SFCO-3k-1mo.csv"

    if not output_csv.exists():
        if not output_tsv.exists():
            if not zip_path.exists():
                if not zip_AA_path.exists():
                    download_file(SFCO_AA_URL, zip_AA_path)
                    print("Zip part A downloaded.")
                if not zip_BB_path.exists():
                    download_file(SFCO_AB_URL, zip_BB_path)
                    print("Zip part B downloaded.")
                with open(zip_path, 'wb') as outfile:
                    for fname in [zip_AA_path, zip_BB_path]:
                        with open(fname, 'rb') as infile:
                            outfile.write(infile.read())
                            print(f"Zip part {fname.name} merged.")
            unzip_file(zip_path, extract_dir)
            print("Zip file extracted.")
            shutil.move(extracted_file, output_tsv)
            shutil.rmtree(extract_dir)
            print("Extracted file moved to the final location.")
        convert_sfco(output_tsv, output_csv, 30)
        print("Data converted to CSV format.")

    print("Preprocessing trajectories...")
    preprocess_trajectories(
        output_csv=output_csv,
        resolutions=cfg.resolutions,
        window_size=cfg.window_size,
        data_dir=sfco_dir,
        global_train_split_str=SFCO_TRAIN_SPLIT,
        daily_split_hour=cfg.daily_split_hour,
        min_inference_activity=cfg.min_inference_activity
    )
    print("Preprocessing complete.")

    print("Processing embeddings and vocab...")
    process_embeddings_and_vocab(
        data_dir=sfco_dir,
        resolutions=cfg.resolutions,
        embedding_dim=cfg.embedding_dim,
        min_inference_activity=cfg.min_inference_activity
    )
    print("Embeddings and vocab processing complete.")

    print("Generating edges...")
    generate_edges(sfco_dir, cfg.resolutions, cfg.window_size, cfg.daily_split_hour)
    print("Edges generation complete.")
    print(generate_stats(sfco_dir, cfg.resolutions, cfg.window_size))
    print("Statistics saved to data directory.")

    if cfg.clean:
        print("Cleaning up intermediate files...")
        cleanup(zip_AA_path, zip_BB_path, zip_path, extract_dir, output_tsv)
        print("Cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Processing Pipeline")
    parser.add_argument('dataset', type=str, help='Chosen dataset (geolife|sfco)')
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory for processed data and intermediate files.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS,
                        help="H3 resolutions to process.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help="Time window size in seconds.")
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="Dimension of the generated embedding vectors. \
                            If not provided, embeddings will not be generated.")
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Clean up intermediate files after processing.")
    parser.add_argument("--min-inference-activity", type=int, default=MIN_INFERENCE_ACTIVITY,
                        help="Minimum number of time windows required for test trajectories.")
    parser.add_argument("--global-train-split", type=str,
                        help="Global split time (YYYYMMDDHHMMSS) \
                            for training vs. test trajectories.")
    parser.add_argument("--daily-split-hour", type=int, default=DAILY_SPLIT_HOUR,
                        help="Hour of the day (0-23) to split \
                            test trajectories (e.g. 11 for 11AM).")
    args = parser.parse_args()

    if args.dataset == "sfco":
        sfco(args)
    elif args.dataset == "geolife":
        geolife(args)
    else:
        print("Invalid dataset selection. Choose geolife or sfco.")
