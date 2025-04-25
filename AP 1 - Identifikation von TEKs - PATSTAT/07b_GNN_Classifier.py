import os
import re
import h5py
import os.path as osp
import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch.nn.functional as F
from torch.nn import Linear, Parameter
import torch_geometric
from torch_geometric.data import HeteroData, Dataset, Data
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import to_networkx
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Base directory for project
BASE_DIR = "/home/thiesen/Documents/Projekt_EDV-TEK/AP 1 - Identifikation von TEKs - PATSTAT"

# Create directories for models and results
MODEL_DIR = os.path.join(BASE_DIR, "models", "gnn")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hardcoded parameters
DATA_PATH = '/mnt/hdd02/Projekt_EDV_TEK/gnn_dataset_identification/edv_tek_identification_gnn_dataset.h5'
HIDDEN_DIM = 256  # Increased from 128 to 256 for better feature representation
BATCH_SIZE = 256  # Reduced from 512 to 256 for better gradient updates
EPOCHS = 50  # Increased from 10 to 50 for better convergence
LEARNING_RATE = 0.001  # Reduced from 0.01 to 0.001 for more stable training
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class PatentHeteroDataset(Dataset):
    def __init__(self, root, data_path, transform=None, pre_transform=None):
        self.data_path = data_path
        super(PatentHeteroDataset, self).__init__(root, transform, pre_transform)
        self.data = None
        self.process()

    @property
    def num_classes(self):
        return 2

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return [osp.basename(self.data_path)]

    @property
    def processed_file_names(self):
        model_name = 'distilbert'  # Default model name
        return f'gnn_tek_data_{model_name}.pt'

    def download(self):
        # Dataset is already downloaded
        pass

    def process(self):
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize HeteroData object
        data = HeteroData()
    
        # Open the HDF5 file
        with h5py.File(self.data_path, 'r') as f:
            print(f"Available datasets in the HDF5 file: {list(f.keys())}")
            
            # Load node features
            # Mapping from our dataset to PyTorch Geometric expected format
            data['patent'].x = torch.tensor(f['patent_nodes/x'][:], dtype=torch.float)
            data['patent'].y = torch.tensor(f['patent_nodes/y'][:], dtype=torch.long)
            data['author'].x = torch.tensor(f['author_nodes/x'][:], dtype=torch.float)
            
            # Load edge indices
            # Patent citations (patent -> patent)
            patent_citations = torch.tensor(f['patent_citations'][:], dtype=torch.long)
            data['patent', 'cites', 'patent'].edge_index = patent_citations.t().contiguous()
            
            # Author-patent edges (author -> patent)
            author_patent_edges = torch.tensor(f['author_patent_edges'][:], dtype=torch.long)
            data['author', 'author_of', 'patent'].edge_index = author_patent_edges.t().contiguous()
            
            # Patent-author edges (patent -> author) - Reverse edges for easier message passing
            patent_author_edges = torch.stack([author_patent_edges[:, 1], author_patent_edges[:, 0]], dim=1)
            data['patent', 'has_author', 'author'].edge_index = patent_author_edges.t().contiguous()

        # Create train, validation, and test masks
        num_patents = data['patent'].num_nodes
        indices = torch.randperm(num_patents)
        
        # Split: 80% train, 10% validation, 10% test
        train_size = int(0.8 * num_patents)
        val_size = int(0.1 * num_patents)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        data['patent'].train_mask = torch.zeros(num_patents, dtype=torch.bool)
        data['patent'].val_mask = torch.zeros(num_patents, dtype=torch.bool)
        data['patent'].test_mask = torch.zeros(num_patents, dtype=torch.bool)
        
        data['patent'].train_mask[train_indices] = True
        data['patent'].val_mask[val_indices] = True
        data['patent'].test_mask[test_indices] = True

        # Print dataset statistics
        print("Dataset Statistics:")
        print(f"Number of patent nodes: {data['patent'].num_nodes}")
        print(f"Number of author nodes: {data['author'].num_nodes}")
        print(f"Number of patent->patent edges: {data['patent', 'cites', 'patent'].edge_index.size(1)}")
        print(f"Number of author->patent edges: {data['author', 'author_of', 'patent'].edge_index.size(1)}")
        print(f"Number of patent->author edges: {data['patent', 'has_author', 'author'].edge_index.size(1)}")
        print(f"Number of training patents: {data['patent'].train_mask.sum().item()}")
        print(f"Number of validation patents: {data['patent'].val_mask.sum().item()}")
        print(f"Number of test patents: {data['patent'].test_mask.sum().item()}")

        self.data = data
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names))

    def len(self):
        return 1

    def get(self, idx):
        return self.data


class SimplifiedHeteroGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features_dict, num_classes):
        super(SimplifiedHeteroGCN, self).__init__()
        torch.manual_seed(42)  # For reproducible results
        
        self.dropout = torch.nn.Dropout(p=0.2)
        self.name = "SimplifiedGNN"

        # Define convolution layers for patent-patent edges only
        self.conv1 = HeteroConv({
            ('patent', 'cites', 'patent'): SAGEConv(num_node_features_dict['patent'], hidden_channels)
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('patent', 'cites', 'patent'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='mean')

        # Linear layer for classification
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # Apply dropout to patent node features
        x_dict['patent'] = self.dropout(x_dict['patent'])

        # First convolution layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Second convolution layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Predictions for 'patent' node embeddings
        out = self.lin(x_dict['patent'])
        return out


class FullHeteroGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features_dict, num_classes):
        super(FullHeteroGCN, self).__init__()
        torch.manual_seed(42)  # For reproducible results
        
        self.name = "FullGNN"
        
        # Define a separate SAGEConv for each edge type
        self.conv1 = HeteroConv({
            ('patent', 'cites', 'patent'): SAGEConv(num_node_features_dict['patent'], hidden_channels),
            ('author', 'author_of', 'patent'): SAGEConv(num_node_features_dict['author'], hidden_channels),
            ('patent', 'has_author', 'author'): SAGEConv(num_node_features_dict['patent'], hidden_channels)
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('patent', 'cites', 'patent'): SAGEConv(hidden_channels, hidden_channels),
            ('author', 'author_of', 'patent'): SAGEConv(hidden_channels, hidden_channels),
            ('patent', 'has_author', 'author'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='mean')

        # Linear layer for classification
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # Apply dropout to patent node features
        x_dict['patent'] = self.dropout(x_dict['patent'])

        # First convolution layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Second convolution layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Predictions for 'patent' node embeddings
        out = self.lin(x_dict['patent'])
        return out


class EnhancedHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features_dict, num_classes):
        super(EnhancedHeteroGNN, self).__init__()
        torch.manual_seed(42)  # For reproducible results
        
        self.name = "EnhancedGNN"
        self.hidden_channels = hidden_channels
        
        # Layer normalization for better training stability
        self.patent_norm = torch.nn.LayerNorm(num_node_features_dict['patent'])
        self.author_norm = torch.nn.LayerNorm(num_node_features_dict['author'])
        
        # Initial feature transformation
        self.patent_lin = torch.nn.Linear(num_node_features_dict['patent'], hidden_channels)
        self.author_lin = torch.nn.Linear(num_node_features_dict['author'], hidden_channels)
        
        # Multiple GAT layers for patent-patent edges with attention
        self.patent_gat1 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.patent_gat2 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        
        # SAGE layers for author-patent edges
        self.author_sage1 = SAGEConv(hidden_channels, hidden_channels)
        self.author_sage2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Layer normalization between layers
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.norm3 = torch.nn.LayerNorm(hidden_channels)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(p=0.3)
        
        # Final classification layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # Initial feature normalization and transformation
        x_dict['patent'] = self.patent_norm(x_dict['patent'])
        x_dict['author'] = self.author_norm(x_dict['author'])
        
        x_dict['patent'] = self.patent_lin(x_dict['patent'])
        x_dict['author'] = self.author_lin(x_dict['author'])
        
        # First layer - Patent GAT
        patent_out = self.patent_gat1(x_dict['patent'], edge_index_dict[('patent', 'cites', 'patent')])
        patent_out = F.relu(patent_out)
        patent_out = self.dropout(patent_out)
        patent_out = self.norm1(patent_out)
        
        # Residual connection for patents
        x_dict['patent'] = patent_out + x_dict['patent']
        
        # First layer - Author SAGE
        author_out = self.author_sage1(x_dict['author'], edge_index_dict[('author', 'author_of', 'patent')])
        author_out = F.relu(author_out)
        author_out = self.dropout(author_out)
        author_out = self.norm2(author_out)
        
        # Residual connection for authors
        x_dict['author'] = author_out + x_dict['author']
        
        # Second layer - Patent GAT
        patent_out = self.patent_gat2(x_dict['patent'], edge_index_dict[('patent', 'cites', 'patent')])
        patent_out = F.relu(patent_out)
        patent_out = self.dropout(patent_out)
        patent_out = self.norm3(patent_out)
        
        # Final residual connection for patents
        x_dict['patent'] = patent_out + x_dict['patent']
        
        # Final classification
        out = self.classifier(x_dict['patent'])
        return out


class TransformerHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features_dict, num_classes):
        super(TransformerHeteroGNN, self).__init__()
        torch.manual_seed(42)  # For reproducible results
        
        self.name = "TransformerGNN"
        self.hidden_channels = hidden_channels
        
        # Feature dimensions with increase for intermediate representations
        expanded_dim = hidden_channels * 2
        
        # Input layer normalization for better training stability
        self.patent_norm = torch.nn.LayerNorm(num_node_features_dict['patent'])
        self.author_norm = torch.nn.LayerNorm(num_node_features_dict['author'])
        
        # Initial feature transformations with larger capacity
        self.patent_lin = torch.nn.Sequential(
            torch.nn.Linear(num_node_features_dict['patent'], expanded_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(expanded_dim),
            torch.nn.Linear(expanded_dim, hidden_channels)
        )
        
        self.author_lin = torch.nn.Sequential(
            torch.nn.Linear(num_node_features_dict['author'], expanded_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(expanded_dim),
            torch.nn.Linear(expanded_dim, hidden_channels)
        )
        
        # Multi-head attention layers for patent-patent relationships
        self.patent_att1 = GATConv(hidden_channels, hidden_channels // 8, heads=8, dropout=0.2, concat=True)
        self.patent_att2 = GATConv(hidden_channels, hidden_channels // 8, heads=8, dropout=0.3, concat=True)
        
        # SAGE layers for author-patent and patent-author relationships
        self.author_patent_sage = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.patent_author_sage = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        
        # Additional convolution for direct patent features
        self.patent_conv = GCNConv(hidden_channels, hidden_channels)
        
        # Layer normalization between layers for stability
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.norm3 = torch.nn.LayerNorm(hidden_channels)
        self.norm4 = torch.nn.LayerNorm(hidden_channels)
        
        # Multi-scale feature aggregation
        self.scale_weights = Parameter(torch.ones(3))
        
        # Adaptive dropouts
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.4)
        self.dropout3 = torch.nn.Dropout(p=0.2)
        
        # Advanced classifier with deeper architecture
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # Initial feature normalization
        x_dict['patent'] = self.patent_norm(x_dict['patent'])
        x_dict['author'] = self.author_norm(x_dict['author'])
        
        # Initial feature transformations
        x_patent_orig = self.patent_lin(x_dict['patent'])
        x_author_orig = self.author_lin(x_dict['author'])
        
        # Store original embeddings for residual connections
        x_dict['patent'] = x_patent_orig
        x_dict['author'] = x_author_orig
        
        # Multi-scale feature collection at different depths
        patent_features = [x_patent_orig]  # Store initial features
        
        # First layer - Patent GAT with attention
        patent_att_out = self.patent_att1(x_dict['patent'], edge_index_dict[('patent', 'cites', 'patent')])
        patent_att_out = self.norm1(patent_att_out)
        patent_att_out = F.gelu(patent_att_out)
        patent_att_out = self.dropout1(patent_att_out)
        
        # Strong residual connection
        x_dict['patent'] = patent_att_out + x_dict['patent']
        patent_features.append(x_dict['patent'])  # Store these features
        
        # First layer - Author to Patent message passing
        author_to_patent = self.author_patent_sage(x_dict['author'], edge_index_dict[('author', 'author_of', 'patent')])
        author_to_patent = self.norm2(author_to_patent)
        author_to_patent = F.gelu(author_to_patent)
        
        # First layer - Patent to Author message passing
        patent_to_author = self.patent_author_sage(x_dict['patent'], edge_index_dict[('patent', 'has_author', 'author')])
        patent_to_author = self.norm3(patent_to_author)
        patent_to_author = F.gelu(patent_to_author)
        
        # Update node representations with messages
        x_dict['patent'] = x_dict['patent'] + 0.5 * author_to_patent
        x_dict['author'] = x_dict['author'] + 0.5 * patent_to_author
        x_dict['patent'] = self.dropout2(x_dict['patent'])
        x_dict['author'] = self.dropout2(x_dict['author'])
        
        # Additional direct patent convolution
        patent_conv_out = self.patent_conv(x_dict['patent'], edge_index_dict[('patent', 'cites', 'patent')][0])
        patent_conv_out = F.gelu(patent_conv_out)
        x_dict['patent'] = x_dict['patent'] + 0.3 * patent_conv_out
        
        # Second layer - Patent GAT
        patent_att_out2 = self.patent_att2(x_dict['patent'], edge_index_dict[('patent', 'cites', 'patent')])
        patent_att_out2 = self.norm4(patent_att_out2)
        patent_att_out2 = F.gelu(patent_att_out2)
        patent_att_out2 = self.dropout3(patent_att_out2)
        
        # Final residual connection
        x_dict['patent'] = x_dict['patent'] + patent_att_out2
        patent_features.append(x_dict['patent'])  # Store these features
        
        # Multi-scale feature aggregation with learned weights
        scale_weights = F.softmax(self.scale_weights, dim=0)
        multi_scale_features = torch.zeros_like(x_dict['patent'])
        for i, features in enumerate(patent_features):
            multi_scale_features += scale_weights[i] * features
        
        # Concatenate multi-scale features with final features for richer representation
        combined_features = torch.cat([multi_scale_features, x_dict['patent']], dim=1)
        
        # Final classification
        out = self.classifier(combined_features)
        return out


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        try:
            out = model(batch)
            mask = batch['patent'].train_mask
            loss = criterion(out[mask], batch['patent'].y[mask])
            loss.backward()
            optimizer.step()
            
            # Collect predictions and labels
            pred = out.argmax(dim=1)
            all_preds.extend(pred[mask].cpu().numpy())
            all_labels.extend(batch['patent'].y[mask].cpu().numpy())
            
            total_loss += loss.item()
            total_batches += 1
        except Exception as e:
            print(f"Error during training: {e}")
            continue
    
    # Calculate metrics
    avg_loss = total_loss / total_batches if total_batches else 0
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
    
    # Only calculate other metrics if we have predictions
    if all_labels:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
    else:
        precision, recall, f1 = 0, 0, 0
            
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def evaluate(model, loader, device, mask_name='test_mask'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating ({mask_name})"):
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            
            mask = getattr(batch['patent'], mask_name)
            labels = batch['patent'].y
            
            all_preds.extend(pred[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    
    if not all_labels:
        return {
            'accuracy': 0,
            'f1': 0,
            'precision': 0,
            'recall': 0
        }
        
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_and_evaluate_model(model_class, model_name, data, device):
    """Train and evaluate a GNN model"""
    print(f"\n{'='*50}")
    print(f"Training model: {model_name}")
    print(f"{'='*50}")
    
    # Get feature dimensions
    num_node_features_dict = {
        'patent': data['patent'].x.size(1),
        'author': data['author'].x.size(1)
    }
    
    # Initialize model
    model = model_class(
        hidden_channels=HIDDEN_DIM,
        num_node_features_dict=num_node_features_dict,
        num_classes=2
    ).to(device)
    
    # Create loaders with just the core parameters that should be compatible across versions
    train_loader = NeighborLoader(
        data, 
        num_neighbors=[20, 10],  # Increased neighborhood size
        batch_size=BATCH_SIZE,
        input_nodes=('patent', data['patent'].train_mask)
    )
    
    val_loader = NeighborLoader(
        data, 
        num_neighbors=[20, 10],  # Increased neighborhood size
        batch_size=BATCH_SIZE,
        input_nodes=('patent', data['patent'].val_mask)
    )
    
    test_loader = NeighborLoader(
        data, 
        num_neighbors=[20, 10],  # Increased neighborhood size
        batch_size=BATCH_SIZE,
        input_nodes=('patent', data['patent'].test_mask)
    )
    
    # Set up optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    best_epoch = 0
    patience = 15  # Early stopping patience
    no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_metrics = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device, 'val_mask')
        val_accs.append(val_metrics['accuracy'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])
        
        # Save the best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            no_improve = 0
            model_path = os.path.join(MODEL_DIR, f"{model.name}_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with validation accuracy: {val_metrics['accuracy']:.4f}")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        print(f'Epoch: {epoch:03d}, Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, f"{model.name}_final.pt")
    torch.save(model, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Load the best model for testing
    try:
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{model.name}_best.pt")))
        print(f"Loaded best model from epoch {best_epoch}")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device, 'test_mask')
    print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}, F1: {test_metrics["f1"]:.4f}')
    
    # Save results
    results = {
        "model_name": model.name,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "early_stopping_patience": patience,
            "lr_scheduler": "ReduceLROnPlateau",
            "optimizer": "AdamW",
            "weight_decay": 0.01
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save detailed results
    with open(os.path.join(RESULTS_DIR, f"{model.name}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    return {
        "model": model.name,
        "test_accuracy": test_metrics["accuracy"],
        "best_val_accuracy": best_val_acc,
        "f1_score": test_metrics["f1"]
    }


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {DATA_PATH}")
    root_dir = os.path.dirname(DATA_PATH)
    dataset = PatentHeteroDataset(root=root_dir, data_path=DATA_PATH)
    data = dataset[0].to(device)
    
    # Models to train
    models_to_train = [
        (SimplifiedHeteroGCN, "Simplified GNN (patent edges only)"),
        (FullHeteroGCN, "Full GNN (all edge types)"),
        (EnhancedHeteroGNN, "Enhanced GNN (patent-patent attention)"),
        (TransformerHeteroGNN, "Transformer GNN (multi-scale features)")
    ]
    
    results_summary = []
    
    # Train and evaluate each model
    for model_class, model_desc in models_to_train:
        try:
            print(f"\nAttempting to train model: {model_desc}")
            # Add version check before training
            print("PyTorch Geometric version:", torch_geometric.__version__)
            
            result = train_and_evaluate_model(model_class, model_desc, data, device)
            results_summary.append(result)
        except Exception as e:
            print(f"Error training model {model_desc}: {str(e)}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            
            # Try to continue with the next model
            continue
    
    # Print summary of results
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Model':<20} {'Test Accuracy':<15} {'Best Val Acc':<15} {'F1 Score':<10}")
    print("-" * 60)
    for result in results_summary:
        print(f"{result['model']:<20} {result['test_accuracy']:<15.4f} {result['best_val_accuracy']:<15.4f} {result['f1_score']:<10.4f}")
    
    # Save overall results
    with open(os.path.join(RESULTS_DIR, "gnn_overall_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\nAll models trained and evaluated. Results saved in {MODEL_DIR} and {RESULTS_DIR} directories.")


if __name__ == "__main__":
    main()