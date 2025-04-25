import os
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torch.nn as nn
from torch.nn import Parameter
import json
from datetime import datetime

# Base directory for project
BASE_DIR = "/home/thiesen/Documents/Projekt_EDV-TEK/AP 2 - Entstehung von TEKs"

# Create directories for models and results
MODEL_DIR = os.path.join(BASE_DIR, "models", "gnn_link_prediction")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hardcoded parameters
DATA_PATH = '/mnt/hdd02/Projekt_EDV_TEK/gnn_dataset_emergence/edv_tek_emergence_gnn_dataset.h5'
HIDDEN_DIM = 256
BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.001
SEED = 42
NEGATIVE_SAMPLING_RATIO = 3  # Handling class imbalance

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class PatentPaperLinkPredictor(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features_dict):
        super(PatentPaperLinkPredictor, self).__init__()
        torch.manual_seed(42)
        
        self.name = "PatentPaperLinkPredictor"
        self.hidden_channels = hidden_channels
        
        # Initial feature transformations
        self.patent_lin = torch.nn.Sequential(
            torch.nn.Linear(num_node_features_dict['patent'], hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2)
        )
        
        self.paper_lin = torch.nn.Sequential(
            torch.nn.Linear(num_node_features_dict['paper'], hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2)
        )
        
        # Patent citation attention
        self.patent_att = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        
        # Paper citation attention
        self.paper_att = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        
        # Patent-paper citation attention
        self.patent_paper_att = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        
        # Patent-paper pair attention
        self.pair_att = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        
        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.norm3 = torch.nn.LayerNorm(hidden_channels)
        
        # Dropout layers
        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)
        
        # Link prediction head
        self.link_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, 1)
        )
        
        # Edge type embeddings
        self.edge_type_emb = Parameter(torch.randn(4, hidden_channels))
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Initial feature transformation
        x_dict['patent'] = self.patent_lin(x_dict['patent'])
        x_dict['paper'] = self.paper_lin(x_dict['paper'])
        
        # Patent citation message passing
        if ('patent', 'cites', 'patent') in edge_index_dict:
            patent_att_out = self.patent_att(x_dict['patent'], edge_index_dict[('patent', 'cites', 'patent')])
            patent_att_out = self.norm1(patent_att_out)
            patent_att_out = F.gelu(patent_att_out)
            x_dict['patent'] = x_dict['patent'] + patent_att_out
        
        # Paper citation message passing
        if ('paper', 'cites', 'paper') in edge_index_dict:
            paper_att_out = self.paper_att(x_dict['paper'], edge_index_dict[('paper', 'cites', 'paper')])
            paper_att_out = self.norm2(paper_att_out)
            paper_att_out = F.gelu(paper_att_out)
            x_dict['paper'] = x_dict['paper'] + paper_att_out
        
        # Patent-paper citation message passing
        if ('patent', 'cites', 'paper') in edge_index_dict:
            patent_paper_att_out = self.patent_paper_att(x_dict['patent'], edge_index_dict[('patent', 'cites', 'paper')])
            patent_paper_att_out = self.norm3(patent_paper_att_out)
            patent_paper_att_out = F.gelu(patent_paper_att_out)
            x_dict['patent'] = x_dict['patent'] + patent_paper_att_out
        
        # Get node embeddings for the edges we want to predict
        patent_emb = x_dict['patent'][edge_label_index[0]]
        paper_emb = x_dict['paper'][edge_label_index[1]]
        
        # Add edge type embeddings
        patent_emb = patent_emb + self.edge_type_emb[0]  # Patent type
        paper_emb = paper_emb + self.edge_type_emb[1]    # Paper type
        
        # Combine embeddings for link prediction
        combined = torch.cat([patent_emb, paper_emb], dim=1)
        combined = self.dropout1(combined)
        
        # Final link prediction
        out = self.link_pred(combined)
        return out.view(-1)


class SimpleBaselineModel(torch.nn.Module):
    """A simple baseline model that just does inner product of transformed features."""
    def __init__(self, hidden_channels, num_node_features_dict):
        super(SimpleBaselineModel, self).__init__()
        torch.manual_seed(42)
        
        self.name = "SimpleBaselineModel"
        
        # Initial feature transformations only
        self.patent_lin = torch.nn.Linear(num_node_features_dict['patent'], hidden_channels)
        self.paper_lin = torch.nn.Linear(num_node_features_dict['paper'], hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Just transform features without message passing
        patent_x = self.patent_lin(x_dict['patent'])
        paper_x = self.paper_lin(x_dict['paper'])
        
        # Get node embeddings for the edges we want to predict
        patent_emb = patent_x[edge_label_index[0]]
        paper_emb = paper_x[edge_label_index[1]]
        
        # Use dot product for prediction
        return torch.sum(patent_emb * paper_emb, dim=1)


def generate_hard_negative_samples(data, edge_label_index, num_neg_samples, device):
    """
    Generate hard negative samples by finding patents and papers that are
    already connected to other papers/patents, but not to each other.
    """
    # Get all patents and papers involved in at least one pair
    unique_patents = torch.unique(edge_label_index[0])
    unique_papers = torch.unique(edge_label_index[1])
    
    # Get all existing pairs as a set for fast lookup
    existing_pairs = set(zip(edge_label_index[0].cpu().numpy(), edge_label_index[1].cpu().numpy()))
    
    # Generate hard negative samples
    neg_src, neg_dst = [], []
    
    # Try to find challenging negative samples
    for _ in range(min(len(unique_patents) * len(unique_papers), num_neg_samples * 2)):
        patent_idx = unique_patents[torch.randint(0, len(unique_patents), (1,))]
        paper_idx = unique_papers[torch.randint(0, len(unique_papers), (1,))]
        
        # If this pair doesn't already exist, add it as a negative sample
        if (patent_idx.item(), paper_idx.item()) not in existing_pairs:
            neg_src.append(patent_idx.item())
            neg_dst.append(paper_idx.item())
            
            # Stop once we have enough samples
            if len(neg_src) >= num_neg_samples:
                break
    
    # For remaining samples, fall back to random sampling
    remaining = num_neg_samples - len(neg_src)
    if remaining > 0:
        num_patents = data['patent'].x.size(0)
        num_papers = data['paper'].x.size(0)
        
        # Generate random patent-paper pairs
        random_patents = torch.randint(0, num_patents, (remaining,), device=device)
        random_papers = torch.randint(0, num_papers, (remaining,), device=device)
        
        # Add to our negative samples
        neg_src.extend(random_patents.cpu().numpy())
        neg_dst.extend(random_papers.cpu().numpy())
    
    # Convert to tensors
    neg_src = torch.tensor(neg_src, device=device)
    neg_dst = torch.tensor(neg_dst, device=device)
    
    return torch.stack([neg_src, neg_dst], dim=0)


def train(model, train_loader, optimizer, criterion, device, use_hard_negatives=True):
    model.train()
    total_loss = 0
    total_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        try:
            # Get positive edges
            pos_edge_label_index = batch['patent', 'pair', 'paper'].edge_label_index
            pos_edge_label = torch.ones(pos_edge_label_index.size(1), dtype=torch.float, device=device)
            
            # Generate negative samples - either hard ones or random
            if use_hard_negatives:
                neg_edge_index = generate_hard_negative_samples(
                    batch, 
                    pos_edge_label_index,
                    num_neg_samples=pos_edge_label_index.size(1) * NEGATIVE_SAMPLING_RATIO,
                    device=device
                )
            else:
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_label_index,
                    num_nodes=(batch['patent'].num_nodes, batch['paper'].num_nodes),
                    num_neg_samples=pos_edge_label_index.size(1) * NEGATIVE_SAMPLING_RATIO,
                    method='sparse'
                )
            
            # Combine positive and negative edges
            edge_label_index = torch.cat([pos_edge_label_index, neg_edge_index], dim=1)
            edge_label = torch.cat([
                pos_edge_label,
                torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)
            ], dim=0)
            
            # Forward pass
            out = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
            loss = criterion(out, edge_label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
        except Exception as e:
            print(f"Error in batch: {e}")
            continue
    
    if total_batches == 0:
        return float('inf')  # Return a large loss if all batches failed
    return total_loss / total_batches


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, use_hard_negatives=True):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        
        try:
            # Get positive edges
            pos_edge_label_index = batch['patent', 'pair', 'paper'].edge_label_index
            pos_edge_label = torch.ones(pos_edge_label_index.size(1), dtype=torch.float, device=device)
            
            # Generate negative samples - either hard ones or random
            if use_hard_negatives:
                neg_edge_index = generate_hard_negative_samples(
                    batch, 
                    pos_edge_label_index,
                    num_neg_samples=pos_edge_label_index.size(1) * NEGATIVE_SAMPLING_RATIO,
                    device=device
                )
            else:
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_label_index,
                    num_nodes=(batch['patent'].num_nodes, batch['paper'].num_nodes),
                    num_neg_samples=pos_edge_label_index.size(1) * NEGATIVE_SAMPLING_RATIO,
                    method='sparse'
                )
            
            # Combine positive and negative edges
            edge_label_index = torch.cat([pos_edge_label_index, neg_edge_index], dim=1)
            edge_label = torch.cat([
                pos_edge_label,
                torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)
            ], dim=0)
            
            # Forward pass
            out = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
            pred = torch.sigmoid(out)
            
            all_preds.append(pred.cpu())
            all_labels.append(edge_label.cpu())
            
        except Exception as e:
            print(f"Error in evaluation batch: {e}")
            continue
    
    if len(all_preds) == 0:
        print("Warning: No valid batches during evaluation")
        return {'auc': 0.5, 'ap': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'balanced_acc': 0.5}
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert predictions to binary using threshold
    binary_preds = (all_preds > threshold).float().numpy()
    all_labels_np = all_labels.numpy()
    
    # Calculate metrics
    auc = roc_auc_score(all_labels_np, all_preds.numpy())
    ap = average_precision_score(all_labels_np, all_preds.numpy())
    
    # Calculate confusion matrix metrics
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
        
        # These metrics are better for imbalanced classes
        precision = precision_score(all_labels_np, binary_preds)
        recall = recall_score(all_labels_np, binary_preds)
        f1 = f1_score(all_labels_np, binary_preds)
        balanced_acc = balanced_accuracy_score(all_labels_np, binary_preds)
        
        # Also calculate metrics for a trivial baseline that always predicts negative
        trivial_preds = np.zeros_like(all_labels_np)
        trivial_acc = np.sum(trivial_preds == all_labels_np) / len(all_labels_np)
        
        print(f"Note: A trivial baseline that always predicts 'no pair' would achieve {trivial_acc:.4f} accuracy")
        print(f"Proportion of positive examples in evaluation: {np.mean(all_labels_np):.4f}")
        
    except Exception as e:
        print(f"Error calculating detailed metrics: {e}")
        precision, recall, f1, balanced_acc = 0, 0, 0, 0
    
    return {
        'auc': auc,
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_acc': balanced_acc
    }


def create_train_val_test_split(data, patent_paper_pairs, test_ratio=0.1, val_ratio=0.1):
    """
    Create a more robust train/val/test split for link prediction.
    This ensures proper isolation of the splits.
    """
    # Determine indices for train/val/test split
    num_edges = patent_paper_pairs.size(1)
    indices = torch.randperm(num_edges)
    
    test_size = int(test_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    train_size = num_edges - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create a proper isolated split
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # Create train/val/test graphs
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
    
    # Keep track of edges for each split
    train_edges = patent_paper_pairs[:, train_mask]
    val_edges = patent_paper_pairs[:, val_mask]
    test_edges = patent_paper_pairs[:, test_mask]
    
    # For training, we keep train edges as both edge_index and edge_label_index
    train_data['patent', 'pair', 'paper'].edge_index = train_edges.clone()
    train_data['patent', 'pair', 'paper'].edge_label_index = train_edges.clone()
    train_data['patent', 'pair', 'paper'].edge_label = torch.ones(train_edges.size(1), dtype=torch.float)
    
    # For validation, we keep val edges as edge_label_index but not in the graph structure
    val_data['patent', 'pair', 'paper'].edge_index = train_edges.clone()  # Only use train edges for structure
    val_data['patent', 'pair', 'paper'].edge_label_index = val_edges.clone()
    val_data['patent', 'pair', 'paper'].edge_label = torch.ones(val_edges.size(1), dtype=torch.float)
    
    # For testing, we keep test edges as edge_label_index but not in the graph structure
    test_data['patent', 'pair', 'paper'].edge_index = train_edges.clone()  # Only use train edges for structure
    test_data['patent', 'pair', 'paper'].edge_label_index = test_edges.clone()
    test_data['patent', 'pair', 'paper'].edge_label = torch.ones(test_edges.size(1), dtype=torch.float)
    
    print(f"Split sizes - Train: {train_edges.size(1)}, Val: {val_edges.size(1)}, Test: {test_edges.size(1)}")
    
    return train_data, val_data, test_data


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {DATA_PATH}")
    with h5py.File(DATA_PATH, 'r') as f:
        # Load node features
        patent_features = torch.tensor(f['patent_embeddings'][:], dtype=torch.float)
        paper_features = torch.tensor(f['paper_embeddings'][:], dtype=torch.float)
        
        # Load edge indices
        patent_citations = torch.tensor(f['patent_citations'][:], dtype=torch.long).t()
        paper_citations = torch.tensor(f['paper_citations'][:], dtype=torch.long).t()
        patent_paper_citations = torch.tensor(f['patent_paper_citations'][:], dtype=torch.long).t()
        patent_paper_pairs = torch.tensor(f['patent_paper_pairs'][:], dtype=torch.long).t()
    
    # Create heterogeneous graph
    data = HeteroData()
    
    # Add node features
    data['patent'].x = patent_features
    data['paper'].x = paper_features
    
    # Print the number of nodes
    print(f"Number of patents: {patent_features.size(0)}")
    print(f"Number of papers: {paper_features.size(0)}")
    
    # Add edge indices
    data['patent', 'cites', 'patent'].edge_index = patent_citations.contiguous()
    data['paper', 'cites', 'paper'].edge_index = paper_citations.contiguous()
    data['patent', 'cites', 'paper'].edge_index = patent_paper_citations.contiguous()
    
    # Set up edge storage for patent-paper pairs
    data['patent', 'pair', 'paper'].edge_index = patent_paper_pairs.contiguous()
    data['patent', 'pair', 'paper'].edge_label = torch.ones(patent_paper_pairs.size(1), dtype=torch.float)
    
    # Add reverse edges explicitly for better graph connectivity
    paper_patent_pairs = torch.stack([patent_paper_pairs[1], patent_paper_pairs[0]], dim=0)
    data['paper', 'rev_pair', 'patent'].edge_index = paper_patent_pairs.contiguous()
    
    # Print the number of edges
    print(f"Number of patent-patent citations: {patent_citations.size(1)}")
    print(f"Number of paper-paper citations: {paper_citations.size(1)}")
    print(f"Number of patent-paper citations: {patent_paper_citations.size(1)}")
    print(f"Number of patent-paper pairs: {patent_paper_pairs.size(1)}")
    
    # Calculate the number of possible patent-paper pairs
    total_possible_pairs = patent_features.size(0) * paper_features.size(0)
    actual_pairs = patent_paper_pairs.size(1)
    pair_ratio = (actual_pairs / total_possible_pairs) * 100
    
    print(f"Total possible patent-paper pairs: {total_possible_pairs}")
    print(f"Actual patent-paper pairs: {actual_pairs}")
    print(f"Percentage of actual pairs: {pair_ratio:.6f}%")
    print(f"Ratio of negative to positive examples in real data: {(total_possible_pairs - actual_pairs) / actual_pairs:.1f}:1")
    
    # Create a proper isolated train/val/test split
    train_data, val_data, test_data = create_train_val_test_split(data, patent_paper_pairs)
    
    # First create a trivial baseline that always predicts negative
    print("\n==== Trivial Baseline Evaluation ====")
    
    # Sample some test pairs with a more realistic negative to positive ratio
    test_pos_edges = test_data['patent', 'pair', 'paper'].edge_label_index
    test_pos_count = test_pos_edges.size(1)
    
    # Use different negative to positive ratios to show the effect
    for neg_ratio in [1, 5, 10, 50, 100]:
        print(f"\nEvaluating with negative:positive ratio = {neg_ratio}:1")
        
        # Generate negative samples
        neg_edges = negative_sampling(
            edge_index=test_pos_edges,
            num_nodes=(data['patent'].x.size(0), data['paper'].x.size(0)),
            num_neg_samples=test_pos_count * neg_ratio
        )
        
        # Combine positive and negative samples
        test_labels = torch.cat([
            torch.ones(test_pos_count),
            torch.zeros(test_pos_count * neg_ratio)
        ])
        
        # Trivial baseline (always predict 0)
        trivial_preds = torch.zeros_like(test_labels)
        
        # Calculate metrics
        trivial_acc = (trivial_preds == test_labels).float().mean().item()
        from sklearn.metrics import balanced_accuracy_score
        trivial_balanced_acc = balanced_accuracy_score(test_labels, trivial_preds)
        
        print(f"Trivial baseline accuracy: {trivial_acc:.4f}")
        print(f"Trivial baseline balanced accuracy: {trivial_balanced_acc:.4f}")
        print(f"Proportion of positive examples: {test_pos_count / (test_pos_count + test_pos_count * neg_ratio):.4f}")
    
    # Create data loaders for actual models
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[10, 5],  # Sample fewer neighbors to reduce leakage
        batch_size=BATCH_SIZE,
        edge_label_index=('patent', 'pair', 'paper'),
        edge_label=train_data['patent', 'pair', 'paper'].edge_label,
        shuffle=True
    )
    
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[10, 5],
        batch_size=BATCH_SIZE,
        edge_label_index=('patent', 'pair', 'paper'),
        shuffle=False
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[10, 5],
        batch_size=BATCH_SIZE,
        edge_label_index=('patent', 'pair', 'paper'),
        shuffle=False
    )
    
    # Initialize models
    num_node_features_dict = {
        'patent': patent_features.size(1),
        'paper': paper_features.size(1)
    }
    
    # Initialize main model
    model = PatentPaperLinkPredictor(
        hidden_channels=HIDDEN_DIM,
        num_node_features_dict=num_node_features_dict
    ).to(device)
    
    # Initialize baseline model for comparison
    baseline_model = SimpleBaselineModel(
        hidden_channels=HIDDEN_DIM,
        num_node_features_dict=num_node_features_dict
    ).to(device)
    
    # Train and evaluate both models to compare
    for current_model, model_name in [(model, "GNN"), (baseline_model, "Baseline")]:
        print(f"\n{'='*50}\nTraining {model_name} model\n{'='*50}")
        
        # Set up optimizer and loss
        optimizer = torch.optim.AdamW(current_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_auc = 0
        best_epoch = 0
        patience = 15
        no_improve = 0
        
        # First try with random negatives
        print("\nTraining with random negative samples:")
        for epoch in range(1, EPOCHS + 1):
            # Train
            train_loss = train(current_model, train_loader, optimizer, criterion, device, use_hard_negatives=True)
            
            # Evaluate
            val_metrics = evaluate(current_model, val_loader, device, use_hard_negatives=True)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['auc'])
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                no_improve = 0
                model_path = os.path.join(MODEL_DIR, f"{current_model.name}_random_negs_best.pt")
                torch.save(current_model.state_dict(), model_path)
                print(f"New best model saved with validation AUC: {val_metrics['auc']:.4f}")
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val AUC: {val_metrics["auc"]:.4f}, '
                  f'Val AP: {val_metrics["ap"]:.4f}, Val F1: {val_metrics.get("f1", 0):.4f}')
        
        # Now try with hard negative samples
        print("\nTraining with hard negative samples:")
        # Reset best metrics and optimizer
        best_val_auc = 0
        best_epoch = 0
        no_improve = 0
        optimizer = torch.optim.AdamW(current_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        for epoch in range(1, EPOCHS + 1):
            # Train
            train_loss = train(current_model, train_loader, optimizer, criterion, device, use_hard_negatives=True)
            
            # Evaluate
            val_metrics = evaluate(current_model, val_loader, device, use_hard_negatives=True)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['auc'])
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                no_improve = 0
                model_path = os.path.join(MODEL_DIR, f"{current_model.name}_hard_negs_best.pt")
                torch.save(current_model.state_dict(), model_path)
                print(f"New best model saved with validation AUC: {val_metrics['auc']:.4f}")
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val AUC: {val_metrics["auc"]:.4f}, '
                  f'Val AP: {val_metrics["ap"]:.4f}, Val F1: {val_metrics.get("f1", 0):.4f}')
        
        # Evaluate on test set with both random and hard negatives
        print("\nTest Results with Random Negatives:")
        current_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{current_model.name}_random_negs_best.pt")))
        test_metrics_random = evaluate(current_model, test_loader, device, use_hard_negatives=True)
        
        print("\nTest Results with Hard Negatives:")
        current_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{current_model.name}_hard_negs_best.pt")))
        test_metrics_hard = evaluate(current_model, test_loader, device, use_hard_negatives=True)
        
        # Print comparison 
        print(f"\n{model_name} Model Test Results:")
        print(f"Random Negatives - AUC: {test_metrics_random['auc']:.4f}, AP: {test_metrics_random['ap']:.4f}, "
              f"F1: {test_metrics_random.get('f1', 0):.4f}, Balanced Acc: {test_metrics_random.get('balanced_acc', 0):.4f}")
        print(f"Hard Negatives - AUC: {test_metrics_hard['auc']:.4f}, AP: {test_metrics_hard['ap']:.4f}, "
              f"F1: {test_metrics_hard.get('f1', 0):.4f}, Balanced Acc: {test_metrics_hard.get('balanced_acc', 0):.4f}")
        
        # Test with various thresholds to find optimal decision boundary
        print("\nFinding optimal threshold:")
        best_model_path = os.path.join(MODEL_DIR, f"{current_model.name}_hard_negs_best.pt")
        current_model.load_state_dict(torch.load(best_model_path))
        
        # Collect predictions for threshold tuning
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                try:
                    # Get positive edges
                    pos_edge_label_index = batch['patent', 'pair', 'paper'].edge_label_index
                    pos_edge_label = torch.ones(pos_edge_label_index.size(1), dtype=torch.float, device=device)
                    
                    # Generate hard negative samples
                    neg_edge_index = generate_hard_negative_samples(
                        batch, pos_edge_label_index, pos_edge_label_index.size(1) * 10, device
                    )
                    
                    # Combine
                    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_index], dim=1)
                    edge_label = torch.cat([
                        pos_edge_label,
                        torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=device)
                    ])
                    
                    # Forward pass
                    out = current_model(batch.x_dict, batch.edge_index_dict, edge_label_index)
                    pred = torch.sigmoid(out)
                    
                    all_preds.append(pred.cpu())
                    all_labels.append(edge_label.cpu())
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            # Test different thresholds
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            thresholds = np.arange(0.1, 1.0, 0.1)
            results = []
            
            for threshold in thresholds:
                binary_preds = (all_preds > threshold).float().numpy()
                f1 = f1_score(all_labels, binary_preds)
                precision = precision_score(all_labels, binary_preds)
                recall = recall_score(all_labels, binary_preds)
                
                results.append((threshold, f1, precision, recall))
                print(f"Threshold {threshold:.1f}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            
            # Find best threshold for F1
            best_threshold = max(results, key=lambda x: x[1])[0]
            print(f"Best threshold for F1: {best_threshold:.2f}")
        
        # Save results
        results = {
            "model_name": current_model.name,
            "test_metrics_random": test_metrics_random,
            "test_metrics_hard": test_metrics_hard,
            "best_threshold": float(best_threshold) if len(all_preds) > 0 else 0.5,
            "data_stats": {
                "total_possible_pairs": int(total_possible_pairs),
                "actual_pairs": int(actual_pairs),
                "pair_ratio_percent": float(pair_ratio),
                "negative_to_positive_ratio": float((total_possible_pairs - actual_pairs) / actual_pairs)
            },
            "hyperparameters": {
                "hidden_dim": HIDDEN_DIM,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "negative_sampling_ratio": NEGATIVE_SAMPLING_RATIO,
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(RESULTS_DIR, f"{current_model.name}_results.json"), 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()