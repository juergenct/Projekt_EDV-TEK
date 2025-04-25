import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import spacy
import nltk
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
from datetime import datetime

# Base directory for project
BASE_DIR = "/home/thiesen/Documents/Projekt_EDV-TEK/AP 1 - Identifikation von TEKs - PATSTAT"
# BASE_DIR = "/fibus/fs1/0f/cyh1826/wt/edv_tek"

# Create directories for models and results
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set script parameters
SEED = 42
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 5
LR = 0.01

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load NLP components
spacy_en = spacy.load('en_core_web_sm')
nltk.download('stopwords', quiet=True)
stop_words = nltk.corpus.stopwords.words('english')

# Text preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return tuple(self.df.iloc[index])

# Neural Network Models
class FNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(FNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_class)
        self.name = "FNN"

    def forward(self, text):
        embedded = self.embedding(text)
        x = F.relu(self.fc1(embedded))
        x = self.dropout(x)
        return self.fc2(x)

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.name = "RNN"

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_input = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input)
        hidden = hidden.squeeze(0)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.name = "LSTM"

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_input = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_input)
        hidden = hidden.squeeze(0)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_class)
        self.name = "CNN"

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)

# Collate functions
def collate_batch_simple(batch):
    """Collate function for FNN and CNN models"""
    label_list, text_list = [], []
    for (_text, _label) in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        
        # Only add samples with non-zero length
        if len(processed_text) > 0:
            label_list.append(_label)
            if len(processed_text) > MAX_SEQ_LENGTH:
                processed_text = processed_text[:MAX_SEQ_LENGTH]
            text_list.append(processed_text)
    
    # Proceed only if there are samples with non-zero length
    if text_list:
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
        return label_list.to(device), text_list.to(device)
    else:
        # Return empty tensors if all samples had zero length
        return torch.tensor([], dtype=torch.int64).to(device), torch.tensor([], dtype=torch.int64).to(device)

def collate_batch_sequence(batch):
    """Collate function for RNN and LSTM models"""
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        
        # Only add samples where the length is greater than 0
        if len(processed_text) > 0:
            if len(processed_text) > MAX_SEQ_LENGTH:
                processed_text = processed_text[:MAX_SEQ_LENGTH]
            label_list.append(_label)
            text_list.append(processed_text)
            lengths.append(len(processed_text))
    
    # Proceed only if there are samples with non-zero length
    if lengths:
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
        lengths = torch.tensor(lengths)
        return label_list.to(device), text_list.to(device), lengths.to(device)
    else:
        # Return empty tensors if all samples had length 0
        return torch.tensor([], dtype=torch.int64).to(device), torch.tensor([], dtype=torch.int64).to(device), torch.tensor([], dtype=torch.int64).to(device)

# Training and evaluation functions
def train_model(model, dataloader, optimizer, criterion, model_type='simple'):
    model.train()
    total_loss = 0
    total_acc = 0
    total_count = 0
    log_interval = 500
    
    for idx, batch in enumerate(dataloader):
        if model_type == 'simple':
            label, text = batch
            if label.size(0) == 0:  # Skip empty batches
                continue
            optimizer.zero_grad()
            predicted_logits = model(text)
        else:  # sequence models (RNN, LSTM)
            label, text, lengths = batch
            if label.size(0) == 0:  # Skip empty batches
                continue
            optimizer.zero_grad()
            predicted_logits = model(text, lengths)
        
        loss = criterion(predicted_logits.squeeze(1), label.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        predicted_labels = (torch.sigmoid(predicted_logits) > 0.5).long()
        total_acc += (predicted_labels.squeeze(1) == label).sum().item()
        total_count += label.size(0)
        total_loss += loss.item() * label.size(0)
        
        if idx % log_interval == 0 and idx > 0:
            print(f'Batch: {idx}/{len(dataloader)} | Loss: {total_loss/total_count:.6f} | Accuracy: {total_acc/total_count:.3f}')
    
    avg_loss = total_loss / total_count if total_count > 0 else float('inf')
    avg_acc = total_acc / total_count if total_count > 0 else 0
    
    return avg_loss, avg_acc

def evaluate_model(model, dataloader, criterion, model_type='simple'):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'simple':
                label, text = batch
                if label.size(0) == 0:  # Skip empty batches
                    continue
                predicted_logits = model(text)
            else:  # sequence models (RNN, LSTM)
                label, text, lengths = batch
                if label.size(0) == 0:  # Skip empty batches
                    continue
                predicted_logits = model(text, lengths)
            
            loss = criterion(predicted_logits.squeeze(1), label.float())
            predicted_labels = (torch.sigmoid(predicted_logits) > 0.5).long()
            
            # Store predictions and labels for detailed metrics
            all_preds.extend(predicted_labels.squeeze(1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            total_acc += (predicted_labels.squeeze(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.item() * label.size(0)
    
    avg_loss = total_loss / total_count if total_count > 0 else float('inf')
    avg_acc = total_acc / total_count if total_count > 0 else 0
    
    # Additional metrics for detailed evaluation
    metrics = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return avg_loss, avg_acc, metrics

def train_and_evaluate(model, train_dataloader, val_dataloader, test_dataloader, 
                       criterion, optimizer, scheduler, epochs, model_type='simple'):
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_model(model, train_dataloader, optimizer, criterion, model_type)
        val_loss, val_acc, _ = evaluate_model(model, val_dataloader, criterion, model_type)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model_path = f"{MODEL_DIR}/{model.name}_best.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
        
        # Update learning rate
        scheduler.step()
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{model.name}_best.pt"))
    test_loss, test_acc, test_metrics = evaluate_model(model, test_dataloader, criterion, model_type)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save final model
    torch.save(model, f"{MODEL_DIR}/{model.name}_final.pt")
    
    # Create classification report
    if len(test_metrics['predictions']) > 0:
        report = classification_report(
            test_metrics['labels'], 
            test_metrics['predictions'],
            output_dict=True
        )
    else:
        report = {}
    
    # Save training history and results
    results = {
        "model_name": model.name,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "classification_report": report,
        "hyperparameters": {
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{RESULTS_DIR}/{model.name}_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    return {
        "model": model.name,
        "test_accuracy": test_acc,
        "best_val_loss": min(val_losses)
    }

def main():
    # File paths
    # cleantech_path = '/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_all_cleantech_title_abstract.parquet'
    cleantech_path = '/mnt/hdd02/Projekt_EDV_TEK/edv_tek_all_cleantech_title_abstract.parquet'
    # non_cleantech_path = '/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_non_cleantech_all_title_abstract.csv'
    non_cleantech_path = '/mnt/hdd02/Projekt_EDV_TEK/edv_tek_non_cleantech_all_title_abstract.csv'
    
    # Load data
    print("Loading data...")
    df_cleantech = pd.read_parquet(cleantech_path)
    df_cleantech = df_cleantech.sample(100000, random_state=SEED)
    df_cleantech['label'] = 1
    
    df_non_cleantech = pd.read_csv(non_cleantech_path)
    df_non_cleantech = df_non_cleantech.sample(100000, random_state=SEED)
    df_non_cleantech['label'] = 0
    
    df_cleantech = df_cleantech[df_cleantech['appln_abstract'] != '']
    df_non_cleantech = df_non_cleantech[df_non_cleantech['appln_abstract'] != '']
    df_cleantech.dropna(subset=['appln_abstract'], inplace=True)
    df_non_cleantech.dropna(subset=['appln_abstract'], inplace=True)
    
    # Combine datasets
    df = pd.concat([df_cleantech, df_non_cleantech], ignore_index=True)
    df = df[['appln_id', 'appln_abstract', 'label']]
    
    # Preprocess text
    print("Preprocessing text...")
    df['appln_abstract'] = df['appln_abstract'].astype(str)
    df['appln_abstract'] = df['appln_abstract'].progress_apply(preprocess_text)
    
    # Rename column for consistency
    df = df.rename(columns={'appln_abstract': 'text'})
    df_torch = df[['text', 'label']].reset_index(drop=True)
    
    # Split data
    train_df, test_df = train_test_split(df_torch, test_size=0.1, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Set up tokenizer and vocabulary
    global tokenizer, text_pipeline, vocab
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    
    # Tokenize training data to build vocabulary
    print("Building vocabulary...")
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(train_df['text']), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    # Define text pipeline
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_df)
    val_dataset = TextClassificationDataset(val_df)
    test_dataset = TextClassificationDataset(test_df)
    
    # Models to train
    models_to_train = [
        {
            "model": FNN(len(vocab), 256, 1),
            "collate_fn": collate_batch_simple,
            "model_type": "simple"
        },
        {
            "model": CNN(len(vocab), 256, 100, [3, 4, 5], 1),
            "collate_fn": collate_batch_simple,
            "model_type": "simple"
        },
        {
            "model": RNN(len(vocab), 256, 128, 1),
            "collate_fn": collate_batch_sequence,
            "model_type": "sequence"
        },
        {
            "model": LSTM(len(vocab), 256, 128, 1),
            "collate_fn": collate_batch_sequence,
            "model_type": "sequence"
        }
    ]
    
    results_summary = []
    
    for model_info in models_to_train:
        model = model_info["model"].to(device)
        print(f"\n{'='*50}")
        print(f"Training {model.name} model")
        print(f"{'='*50}")
        
        # Create dataloaders with appropriate collate function
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            collate_fn=model_info["collate_fn"]
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            collate_fn=model_info["collate_fn"]
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            collate_fn=model_info["collate_fn"]
        )
        
        # Setup training components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        
        # Train and evaluate
        result = train_and_evaluate(
            model, 
            train_dataloader, 
            val_dataloader, 
            test_dataloader, 
            criterion, 
            optimizer, 
            scheduler, 
            EPOCHS, 
            model_info["model_type"]
        )
        
        results_summary.append(result)
    
    # Print summary of results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Model':<10} {'Test Accuracy':<15} {'Best Val Loss':<15}")
    print("-" * 50)
    for result in results_summary:
        print(f"{result['model']:<10} {result['test_accuracy']:<15.4f} {result['best_val_loss']:<15.4f}")
    
    # Save overall results
    with open(f"{RESULTS_DIR}/overall_results.json", 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\nAll models trained and evaluated. Results saved in {MODEL_DIR} and {RESULTS_DIR} directories.")

if __name__ == "__main__":
    main()