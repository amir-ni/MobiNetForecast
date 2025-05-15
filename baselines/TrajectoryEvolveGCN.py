import os
from typing import Tuple, Set, List
from tqdm import tqdm

import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric_temporal.nn.recurrent import EvolveGCNH

class Snapshot:
    """
    Snapshot represents a graph at a specific time step.

    Attributes:
        x (torch.Tensor): Node feature matrix (num_nodes x feature_dim).
        edge_index (torch.Tensor): Tensor of shape [2, num_edges] containing edge indices.
        edge_attr (torch.Tensor): Edge attributes.
        edge_set (set): Set of edge tuples for quick lookup.
    """
    def __init__(self,
                 x: torch.Tensor,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 edge_set: Set[Tuple[int, int]]) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_set = edge_set

class EvolveGCNHLinkPredictor(nn.Module):
    """
    EvolveGCNHLinkPredictor leverages historical graph snapshots to predict future links.

    It uses the EvolveGCNH recurrent graph convolution layer to evolve node embeddings over time,
    and an MLP to predict the probability of link existence between two nodes based on the concatenation
    of their embeddings.

    Attributes:
        recurrent (EvolveGCNH): Recurrent graph convolution layer that evolves node embeddings.
        mlp (nn.Sequential): Multi-layer perceptron that predicts link probability from concatenated node embeddings.
    """
    def __init__(self,
                 node_count: int,
                 node_features: int,
                 mlp_hidden_dim: int = 32) -> None:
        super(EvolveGCNHLinkPredictor, self).__init__()
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute link prediction logits for given edge pairs.

        Args:
            x (torch.Tensor): Node feature matrix from the historical snapshot.
            edge_index (torch.Tensor): Edge index tensor from the historical snapshot.
            edge_attr (torch.Tensor): Edge attributes from the historical snapshot.
            edge_pairs (torch.Tensor): Tensor of shape [2, num_edges] for which to predict link existence.

        Returns:
            torch.Tensor: 1D tensor of logits corresponding to each edge pair.
        """
        h = self.recurrent(x, edge_index, edge_attr)
        u_embed = h[edge_pairs[0]]
        v_embed = h[edge_pairs[1]]
        edge_feat = torch.cat([u_embed, v_embed], dim=1)
        logits = self.mlp(edge_feat)
        return logits.squeeze()

class TrajectoryEvolveGCN:
    """
    Wrapper for EvolveGCNH that follows the interface expected by the main training pipeline.
    """
    def __init__(self, config):
        self.config = config
        self.node2idx = None
        self.num_nodes = 0
        self.feature_dim = config.get("n_embd", 16)
        self.mlp_hidden_dim = config.get("mlp_hidden_dim", 32)
        self.threshold = config.get("threshold", 0.5)
        self.num_epochs = config.get("max_epochs", 20)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = config["device"]
        self.logger = None

    def prepare_snapshots(self, dataset):
        """
        Convert trajectory data to graph snapshots.
        """
        # Extract unique locations as nodes
        node2idx_directory = os.path.join(
            self.config["data_dir"],
            self.config["dataset"],
            f"resolution-{self.config['resolution']}",
            "neighbors.json"
        )
        with open(node2idx_directory, encoding='utf-8') as node2idx_file:
            self.node2idx = json.load(node2idx_file)
        self.num_nodes = len(self.node2idx)

        snapshots: List[Snapshot] = []
        for _, group in sorted(dataset.groupby("time"), key=lambda x: x[0]):
            u_list: List[int] = group["user1"].map(lambda x: self.node2idx[x]).tolist()
            v_list: List[int] = group["user2"].map(lambda x: self.node2idx[x]).tolist()
            if len(u_list) == 0:
                continue
            edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
            edge_attr = torch.ones(edge_index.size(1), dtype=torch.float)
            x = torch.randn(self.num_nodes, self.feature_dim)
            edge_set: Set[Tuple[int, int]] = set()
            for u, v in zip(u_list, v_list):
                edge_set.add((u, v))
            snapshots.append(Snapshot(x, edge_index, edge_attr, edge_set))
        return snapshots

    def negative_sampling(self, pos_edge_set, num_neg_samples):
        """
        Randomly sample negative edges (i.e. node pairs not in pos_edge_set) until we have num_neg_samples.

        Args:
            pos_edge_set (set): Set of positive edges for quick membership checking.
            num_neg_samples (int): Number of negative samples to generate.

        Returns:
            torch.Tensor: Tensor of shape [2, num_neg_samples] containing negative edges.
        """
        neg_edges: List[Tuple[int, int]] = []
        while len(neg_edges) < num_neg_samples:
            u = torch.randint(0, self.num_nodes, (1,)).item()
            v = torch.randint(0, self.num_nodes, (1,)).item()
            if u == v:
                continue
            if (u, v) in pos_edge_set or (v, u) in pos_edge_set:
                continue
            neg_edges.append((u, v))
        return torch.tensor(neg_edges, dtype=torch.long).t()

    def train(self, dataset, logger, model_checkpoint_directory):
        """Train the EvolveGCN model on trajectory data."""
        self.logger = logger
        logger.info("Preparing graph snapshots from trajectory data...")

        snapshots = self.prepare_snapshots(dataset)
        if len(snapshots) < 2:
            logger.error("Not enough snapshots for training (need at least 2).")
            return

        logger.info(f"Created {len(snapshots)} graph snapshots with {self.num_nodes} nodes.")

        # Initialize the model
        self.model = EvolveGCNHLinkPredictor(
            node_count=self.num_nodes,
            node_features=self.feature_dim,
            mlp_hidden_dim=self.mlp_hidden_dim
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        logger.info("Starting training...")
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            snapshot_count = 0

            # Use snapshot at time t to predict links at time t+1
            for i in tqdm(range(len(snapshots) - 1), desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                current_snap = snapshots[i]
                target_snap = snapshots[i+1]
                pos_edge_index = target_snap.edge_index
                num_pos = pos_edge_index.size(1)

                neg_edge_index = self.negative_sampling(target_snap.edge_set, num_pos)

                # Combine positive and negative edge pairs
                all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                # Labels: 1 for positive, 0 for negative
                labels = torch.cat([torch.ones(num_pos), torch.zeros(num_pos)], dim=0)

                # Move tensors to the device
                current_x = current_snap.x.to(self.device)
                current_edge_index = current_snap.edge_index.to(self.device)
                current_edge_attr = current_snap.edge_attr.to(self.device)
                all_edge_index = all_edge_index.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(current_x, current_edge_index, current_edge_attr, all_edge_index)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                snapshot_count += 1

            avg_loss = epoch_loss / max(1, snapshot_count)
            logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        logger.info("Training completed.")

        # Save the model
        self.save_checkpoint(model_checkpoint_directory)

    def save_checkpoint(self, model_checkpoint_directory):
        """Save model checkpoint."""
        os.makedirs(model_checkpoint_directory, exist_ok=True)
        checkpoint_path = os.path.join(model_checkpoint_directory, 'checkpoint.pt')

        checkpoint = {
            "epoch": self.num_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "node2idx": self.node2idx,
            "num_nodes": self.num_nodes,
            "config": self.config
        }

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Model saved to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def load_state_dict(self, state_dict):
        """Load model state from checkpoint."""
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.node2idx = state_dict.get("node2idx", self.node2idx)
        self.num_nodes = state_dict.get("num_nodes", self.num_nodes)

    def evaluate(self, dataset, logger):
        """Evaluate the model on test data."""
        self.logger = logger
        logger.info("Preparing graph snapshots from test data...")

        test_snapshots = self.prepare_snapshots(dataset)
        if len(test_snapshots) < 2:
            logger.error("Not enough test snapshots for evaluation (need at least 2).")
            return ["Insufficient test data"]

        logger.info(f"Created {len(test_snapshots)} test graph snapshots.")

        # Evaluation logic
        self.model.eval()

        overall_true_positives = 0
        overall_predicted_edges = 0
        overall_ground_truth_edges = 0

        with torch.no_grad():
            for i in tqdm(range(len(test_snapshots) - 1), desc="Evaluating"):
                current_snap = test_snapshots[i]
                target_snap = test_snapshots[i+1]

                pos_edge_index = target_snap.edge_index
                num_pos = pos_edge_index.size(1)
                neg_edge_index = self.negative_sampling(target_snap.edge_set, num_pos)
                all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                # Move tensors to device
                current_x = current_snap.x.to(self.device)
                current_edge_index = current_snap.edge_index.to(self.device)
                current_edge_attr = current_snap.edge_attr.to(self.device)
                all_edge_index = all_edge_index.to(self.device)

                logits = self.model(current_x, current_edge_index, current_edge_attr, all_edge_index)
                predictions = (torch.sigmoid(logits) >= self.threshold).float()

                # Calculate metrics
                true_positive = (predictions[:num_pos] == 1).sum().item()
                predicted_positive = (predictions == 1).sum().item()

                overall_true_positives += true_positive
                overall_predicted_edges += predicted_positive
                overall_ground_truth_edges += num_pos

        # Calculate final metrics
        false_positives = overall_predicted_edges - overall_true_positives
        false_negatives = overall_ground_truth_edges - overall_true_positives

        precision = overall_true_positives / max(1, overall_true_positives + false_positives)
        recall = overall_true_positives / max(1, overall_true_positives + false_negatives)
        f1_score = 2 * precision * recall / max(0.0001, precision + recall)

        results = [
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1 Score: {f1_score:.4f}",
            f"True Positives: {overall_true_positives}",
            f"False Positives: {false_positives}",
            f"False Negatives: {false_negatives}",
            f"Predicted Edges: {overall_predicted_edges}",
            f"Ground Truth Edges: {overall_ground_truth_edges}"
        ]

        logger.info(", ".join(results))

        # If config requests storing predictions, create edge CSV
        if self.config.get("store_predictions", False):
            self.store_predictions(test_snapshots, logger)

        return results

    def store_predictions(self, test_snapshots, logger):
        """Store predictions for collision detection."""
        predicted_edges = []

        with torch.no_grad():
            for i in range(len(test_snapshots) - 1):
                current_snap = test_snapshots[i]

                # Create all possible pairs to predict on
                all_pairs = []
                for u in range(self.num_nodes):
                    for v in range(u+1, self.num_nodes):
                        all_pairs.append((u, v))

                if not all_pairs:
                    continue

                pairs_tensor = torch.tensor(all_pairs, dtype=torch.long).t()

                # Move tensors to device
                current_x = current_snap.x.to(self.device)
                current_edge_index = current_snap.edge_index.to(self.device)
                current_edge_attr = current_snap.edge_attr.to(self.device)
                pairs_tensor = pairs_tensor.to(self.device)

                # Batch predictions if too many pairs
                batch_size = 10000
                for j in range(0, pairs_tensor.size(1), batch_size):
                    batch_pairs = pairs_tensor[:, j:j+batch_size]
                    batch_logits = self.model(current_x, current_edge_index, current_edge_attr, batch_pairs)
                    batch_preds = (torch.sigmoid(batch_logits) >= self.threshold).cpu().numpy()

                    # Record predicted positive edges
                    for k, (u, v) in enumerate(zip(batch_pairs[0].cpu().numpy(), batch_pairs[1].cpu().numpy())):
                        if batch_preds[k]:
                            # Map indices back to original node IDs
                            idx_to_node = {idx: node for node, idx in self.node2idx.items()}
                            user1 = idx_to_node.get(u, f"unknown_{u}")
                            user2 = idx_to_node.get(v, f"unknown_{v}")
                            time_window = i + 1  # next time step
                            predicted_edges.append((user1, user2, time_window))

        # Save predictions
        if predicted_edges:
            pred_df = pd.DataFrame(predicted_edges, columns=["user1", "user2", "time"])
            pred_path = os.path.join(logger.log_directory, "predicted_edges.csv")
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"Saved {len(predicted_edges)} predicted edges to {pred_path}")
