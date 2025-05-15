import os
import time
import warnings
import itertools
from typing import Dict, List, Optional
from logging import Logger
from tqdm import tqdm
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from MobiNetForecast.TrajectoryBatchDataset import TrajectoryBatchDataset


def calculate_bleu(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the cumulative BLEU score for a batch of predicted and target sequences.

    Args:
        predictions (torch.Tensor): A tensor containing the predicted sequences.
        targets (torch.Tensor): A tensor containing the reference sequences.

    Returns:
        float: The cumulative BLEU score for the batch.
    """
    bleu_score = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for prediction, target in zip(predictions, targets):
            prediction = prediction.tolist()
            target = target.tolist()
            if len(prediction) > 0:
                bleu_score += sentence_bleu([target], prediction)
    return bleu_score


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: TrajectoryBatchDataset,
    config: Dict,
    logger: Logger,
    top_k: Optional[List] = None,
) -> List:
    """
    Evaluates the given model on the provided dataset and configuration.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataset (TrajectoryBatchDataset): The dataset containing trajectories for evaluation.
        config (Dict): A dictionary containing configuration parameters.
        logger (Logger): A logger instance for logging evaluation results.
        top_k (Optional[List]): A list of top-k values for evaluation metrics. Defaults to None.

    Returns:
        list: A list containing evaluation results.
    """
    model.eval()
    device = config["device"]
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    prediction_length = config["test_prediction_length"]
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float32)

    if top_k is None:
        top_k = [1, 3, 5]

    beam_width = config["beam_width"]

    if config["continuity"]:
        neighbors = dataset.get_neighbors()

    total_bleu_score = 0.0
    correct_predictions = {k: torch.zeros(
        prediction_length, dtype=torch.int32).to(device) for k in top_k}

    if config["store_predictions"]:
        predictions_data = []
        pred_results_path = os.path.join(logger.log_directory, "predictions.csv")

    start_time = time.time()

    total_samples = 0
    for X, Y in (p_bar := tqdm(dataset, leave=False)):
        x, y = X.to(device), Y.to(device)
        beams = torch.zeros((x.shape[0], 1, 0), dtype=torch.int32).to(device)
        scores = torch.zeros((x.shape[0], 1), dtype=torch.float32).to(device)

        total_samples += x.shape[0]
        for j in range(prediction_length):
            new_scores, new_beams = [], []
            for b in range(beams.shape[1]):
                beam = beams[:, b:b+1]
                with ctx:
                    input_sequence = torch.cat((x, beam.squeeze(1)), dim=1)
                    logits, _ = model(
                        input_sequence[:, -config["block_size"]:])
                    logits = torch.squeeze(logits, dim=1)
                    logits[:, 0] = float('-inf')
                    probs = torch.softmax(logits, dim=1)

                    # Apply the mask
                    if config["continuity"]:
                        last_prediction = input_sequence[:, -1]
                        mask = torch.zeros_like(logits, dtype=torch.bool)
                        for idx, item in enumerate(last_prediction):
                            mask[idx, neighbors[item.item()]] = True
                        probs[~mask] = 0

                    # Get top-k probabilities and their indices
                    top_probs, indices = torch.topk(probs, beam_width)
                    # Append new indices to beam
                    new_beam = torch.cat(
                        (beam.repeat(1, beam_width, 1), indices.unsqueeze(2)), dim=2)
                    new_score = scores[:, b:b+1] + \
                        torch.log(top_probs)  # Update scores
                    new_scores.append(new_score)
                    new_beams.append(new_beam)
            # Concatenate along beam dimension
            new_scores = torch.cat(new_scores, dim=1)
            # Concatenate along beam dimension
            new_beams = torch.cat(new_beams, dim=1)
            top_scores, top_beams = torch.topk(new_scores.view(
                X.shape[0], -1), beam_width)  # Reshape scores to 2D and get top-k
            # Reshape beams to 3D for gathering
            beams = new_beams.view(X.shape[0], -1, new_beams.shape[2])
            beams = torch.gather(beams, 1, top_beams.unsqueeze(
                2).expand(-1, -1, beams.shape[2]))  # Gather the top-k beams
            scores = top_scores  # Update scores with top-k scores

            for k in correct_predictions.keys():
                if beam_width >= k:
                    predictions = beams[:, :k]  # Get the top-k beams
                    for beam_number in range(k):
                        correct_predictions[k][j] += (((predictions[:, beam_number] == y[:, :j+1]) | (y[:, :j+1] == 0)).all(dim=1).int()).sum().item()

        total_bleu_score += calculate_bleu(beams[:, 0], y)

        if config["store_predictions"]:
            beams_np = beams[:, 0].cpu().numpy()
            batch_indices = dataset.batches[p_bar.n]
            for i, j in enumerate(batch_indices):
                predictions_data.append({
                    "UserID": dataset.test_df.iloc[j]["UserID"],
                    "TrajectoryID": dataset.test_df.iloc[j]["TrajectoryID"],
                    "StartTime": dataset.test_df.iloc[j]["StartTime"],
                    "InputLength": len(dataset.test_df.iloc[j]["InputSequence"].split()),
                    "PredictionSequence": dataset.test_df.iloc[j]["PredictionSequence"],
                    "PredictedSequence": " ".join(map(str, beams_np[i]))
                })

    if config["store_predictions"]:
        pd.DataFrame(predictions_data).to_csv(pred_results_path, index=False)
        logger.info(f"Saved predictions to {pred_results_path}")

    test_duration = time.time() - start_time

    avg_bleu_score = total_bleu_score / total_samples
    acc = {k: (100.0 * v) / (total_samples)
           for k, v in correct_predictions.items()}
    results_dict = {'Dataset': config['dataset']}

    for k, v in acc.items():
        results_dict[f"Acc@{k}"] = round(v[-1].item(), 4)
    results_dict["BLEU"] = round(avg_bleu_score, 4)
    results_dict["Test duration"] = round(test_duration, 3)
    results_dict["Samples"] = total_samples
    for k, v in acc.items():
        results_dict[f"Accuracy@{k} Steps"] = ' '.join([str(round(i.item(), 4)) for i in v])

    logger.info(", ".join([f"{key}: {value}" for key, value in results_dict.items()]))
    return results_dict


def generate_collision_edges(
    config: Dict,
    logger: Logger
) -> None:
    """
    Reads predicted trajectories from 'predictions.csv',
    processes them into collision edges,
    saves these to 'edges.csv'.

    Parameters:
        config (Dict): Contains keys 'window_size', 'data_dir', 'dataset', and 'resolution'.
        logger (Logger): Logger for output messages and log directory.
    """
    try:
        window_size = config["window_size"]
    except KeyError:
        logger.error("Config missing required key: 'window_size'")
        return

    pred_results_path = os.path.join(logger.log_directory, "predictions.csv")
    try:
        df_test = pd.read_csv(pred_results_path)
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"File I/O error writing CSV to {pred_results_path}: {e}")
        return
    except OSError as e:
        logger.error(f"OS error writing CSV to {pred_results_path}: {e}")
        return

    predicted_points = []

    for idx, row in df_test.iterrows():
        try:
            user = row["UserID"]
            start_time = int(row["StartTime"])
            input_length = row["InputLength"]
            predicted_sequence = str(row["PredictedSequence"]).split()
            actual_sequence = len(row["PredictionSequence"].split())
        except KeyError as e:
            logger.error(f"Row {idx}: missing column {e}")
            continue
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Row {idx}: bad data ({e})")
            continue
        for i, cell in enumerate(predicted_sequence[:actual_sequence]):
            point_ts = start_time + (input_length + i) * window_size
            time_window = (point_ts // window_size) + 1
            predicted_points.append((user, time_window, cell))

    predicted_points_df = pd.DataFrame(predicted_points, columns=["UserID", "time", "location"])
    predicted_edges = []
    grouped = predicted_points_df.groupby(["time", "location"])["UserID"]\
      .apply(list).reset_index()
    for _, row in grouped.iterrows():
        users = row["UserID"]
        if len(users) > 1:
            for pair in itertools.combinations(sorted(set(users)), 2):
                predicted_edges.append({
                    "user1": pair[0],
                    "user2": pair[1],
                    "time": row["time"]
                })

    predicted_edges = pd.DataFrame(predicted_edges, columns=["user1", "user2", "time"])
    output_path = os.path.join(logger.log_directory, "edges.csv")
    try:
        predicted_edges.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predicted_edges)} collisions to {output_path}")
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"File I/O error writing CSV to {output_path}: {e}")
        return
    except OSError as e:
        logger.error(f"OS error writing CSV to {output_path}: {e}")
        return


def evaluate_collision_prediction(
    config: Dict,
    logger: Logger,
    generate_edges: bool = True
) -> Dict:
    """
    Evaluate collision predictions against ground truth.

    Reads predicted collisions from 'predictions.csv', processes them into collision edges,
    saves these to 'edges.csv', and compares them with ground truth edges loaded from a CSV.
    Computes and logs precision, recall, F1 score, and related counts.

    Parameters:
        config (Dict): Contains keys 'window_size', 'data_dir', 'dataset', and 'resolution'.
        logger (Logger): Logger for output messages and log directory.

    Returns:
        Dict: Evaluation metrics including precision, recall, F1 score.
    """
    if generate_edges:
        generate_collision_edges(config, logger)

    edges_path = os.path.join(logger.log_directory, "edges.csv")
    try:
        predicted_edges = pd.read_csv(edges_path)
    except FileNotFoundError as e:
        logger.error(f"CSV file not found at {edges_path}: {e}")
        return

    ground_truth_edges_path = os.path.join(
        config["data_dir"],
        config["dataset"],
        f"resolution-{config['resolution']}",
        "edges_test.csv"
    )
    try:
        ground_truth_edges = pd.read_csv(ground_truth_edges_path,\
            usecols=['user1', 'user2', 'time', 'seen'])
    except (FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            OSError) as e:
        logger.error(f"Failed to read ground truth edges CSV at "
                    f"{ground_truth_edges_path}: {e}")
        return

    ground_truth_edges = ground_truth_edges[~ground_truth_edges['seen']].drop(
        columns='seen', errors='ignore').reindex(
        columns=['user1', 'user2', 'time']
    )

    try:
        if len(predicted_edges) == 0:
            logger.warning("No predicted edges found.")
            true_predictions = pd.DataFrame(columns=['user1', 'user2', 'time'])
            return {
                "micro_precision": 0,
                "micro_recall":    0,
                "micro_f1":        0,
                "macro_precision": 0,
                "macro_recall":    0,
                "macro_f1":        0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": len(ground_truth_edges),
                "predicted_collisions": 0,
                "ground_truth_collisions": len(ground_truth_edges)
            }
        else:
            true_predictions = pd.merge(
                predicted_edges, ground_truth_edges,
                how='inner', on=['user1', 'user2', 'time']
            )
    except pd.errors.MergeError as e:
        logger.error(f"MergeError merging predicted and ground-truth edges: {e}")
        return
    except KeyError as e:
        logger.error(f"Missing column during merge: {e}")
        return
    except ValueError as e:
        logger.error(f"ValueError merging edges (perhaps dtype mismatch?): {e}")
        return

    SECONDS_PER_DAY = 24 * 60 * 60


    def add_day_column(df):
        df["day"] = (
            df["time"].astype(int)
            .mul(config["window_size"])
            .floordiv(SECONDS_PER_DAY)
        )
        df["start_window"] = (
            df["day"].astype(int)
            .mul(SECONDS_PER_DAY)
            .add(13 * 60 * 60)
            .floordiv(300)
            .add(1)
        )
        df["window"] = df["time"] - df["start_window"]

    for df in (true_predictions, predicted_edges, ground_truth_edges):
        add_day_column(df)

    tp_by_day = true_predictions .groupby("day").size()
    pred_by_day = predicted_edges   .groupby("day").size()
    gt_by_day   = ground_truth_edges.groupby("day").size()

    all_days = sorted(set(tp_by_day.index)
                    | set(pred_by_day.index)
                    | set(gt_by_day.index))

    per_day = []
    for day in all_days:
        tp = tp_by_day.get(day, 0)
        fp = pred_by_day.get(day, 0) - tp
        fn = gt_by_day.get(day, 0)   - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1

        per_day.append({"day": day, "precision": p, "recall": r, "f1": f1})

    metrics_df = pd.DataFrame(per_day).set_index("day")

    macro_precision = metrics_df["precision"].mean()
    macro_recall    = metrics_df["recall"].mean()
    macro_f1        = metrics_df["f1"].mean()
    tp_total = len(true_predictions)
    fp_total = len(predicted_edges)   - tp_total
    fn_total = len(ground_truth_edges) - tp_total

    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total)>0 else 1
    micro_recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total)>0 else 1
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
                        if (micro_precision + micro_recall)>0 else 1

    results_dict = {
        "micro_precision": micro_precision,
        "micro_recall":    micro_recall,
        "micro_f1":        micro_f1,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "macro_f1":        macro_f1,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "predicted_collisions": len(predicted_edges),
        "ground_truth_collisions": len(ground_truth_edges)
    }

    tp_by_window = true_predictions .groupby("window").size()
    pred_by_window = predicted_edges   .groupby("window").size()
    gt_by_window   = ground_truth_edges.groupby("window").size()

    all_windows = sorted(set(tp_by_window.index)
                    | set(pred_by_window.index)
                    | set(gt_by_window.index))

    per_window = []
    for window in all_windows:
        tp = tp_by_window.get(window, 0)
        fp = pred_by_window.get(window, 0) - tp
        fn = gt_by_window.get(window, 0)   - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1

        per_window.append({"window": window, "precision": p, "recall": r,
                            "f1": f1, "tp": tp, "fp": fp, "fn": fn})

    output_path = os.path.join(logger.log_directory, "windows.csv")
    pd.DataFrame(per_window).sort_values(by=['window']).to_csv(output_path, index=False)

    logger.info(", ".join([f"{key}: {value}" for key, value in results_dict.items()]))
    return results_dict
