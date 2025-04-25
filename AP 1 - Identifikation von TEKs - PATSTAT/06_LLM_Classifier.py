import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
tqdm.pandas()
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import torch

# Base directory for project
BASE_DIR = "/home/thiesen/Documents/Projekt_EDV-TEK/AP 1 - Identifikation von TEKs - PATSTAT"
# BASE_DIR = "/fibus/fs1/0f/cyh1826/wt/edv_tek"

# Create directories for models and results
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Define models to train
models_to_train = [
    "distilbert/distilbert-base-uncased",
    "distilbert/distilroberta-base",
]

def load_data():
    """Load and prepare the dataset."""
    print("Loading data...")
    
    # File paths
    # cleantech_path = '/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_all_cleantech_title_abstract.parquet'
    cleantech_path = '/mnt/hdd02/Projekt_EDV_TEK/edv_tek_all_cleantech_title_abstract.parquet'
    # non_cleantech_path = '/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_non_cleantech_all_title_abstract.csv'
    non_cleantech_path = '/mnt/hdd02/Projekt_EDV_TEK/edv_tek_non_cleantech_all_title_abstract.csv'
    
    # Load positive examples (cleantech)
    df_cleantech = pd.read_parquet(cleantech_path)
    df_cleantech = df_cleantech.sample(100000, random_state=SEED)  # Using smaller dataset like in notebook
    df_cleantech['label'] = 1
    
    # Load negative examples (non-cleantech)
    df_non_cleantech = pd.read_csv(non_cleantech_path)
    df_non_cleantech = df_non_cleantech.sample(100000, random_state=SEED)
    df_non_cleantech['label'] = 0
    
    # Filter empty abstracts and handle nulls
    df_cleantech = df_cleantech[df_cleantech['appln_abstract'] != '']
    df_non_cleantech = df_non_cleantech[df_non_cleantech['appln_abstract'] != '']
    df_cleantech.dropna(subset=['appln_abstract'], inplace=True)
    df_non_cleantech.dropna(subset=['appln_abstract'], inplace=True)
    
    # Combine datasets
    df = pd.concat([df_cleantech, df_non_cleantech], ignore_index=True)
    
    # Handle list-type abstracts
    df['appln_abstract'] = df['appln_abstract'].apply(
        lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x
    )
    
    # Rename column for consistency
    df.rename(columns={'appln_abstract': 'text'}, inplace=True)
    df = df[['text', 'label']]
    
    # Shuffle data
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_datasets(df):
    """Convert DataFrame to Hugging Face datasets and split."""
    dataset = Dataset.from_pandas(df)
    
    # Split into train/test
    train_test_dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    
    # Split train into train/validation
    train_val_dataset = train_test_dataset['train'].train_test_split(test_size=0.1, seed=SEED)
    
    # Combine into a DatasetDict
    final_datasets = DatasetDict({
        'train': train_val_dataset['train'],
        'validation': train_val_dataset['test'],
        'test': train_test_dataset['test']
    })
    
    print(f"Train size: {len(final_datasets['train'])}")
    print(f"Validation size: {len(final_datasets['validation'])}")
    print(f"Test size: {len(final_datasets['test'])}")
    
    return final_datasets

def train_evaluate_model(model_name, datasets):
    """Train and evaluate a model."""
    print(f"\n{'='*50}")
    print(f"Training model: {model_name}")
    print(f"{'='*50}")
    
    # Create model-specific directories
    model_short_name = model_name.split('/')[-1]
    model_save_dir = os.path.join(MODEL_DIR, model_short_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    try:
        # Load tokenizer and model
        print(f"Loading tokenizer and model for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        print("Please check if the model name is correct and accessible.")
        raise e
    
    # Define preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(preprocess_function, batched=True, num_proc=4)  # Use multiple processes for faster tokenization
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=16,  # Smaller batch size for memory efficiency
        num_train_epochs=5,  # Reduced from 10 for faster training
        weight_decay=0.01,
        evaluation_strategy='epoch',
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
    )
    
    # Define metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train model
    print("Starting training...")
    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Save partial results
        with open(os.path.join(RESULTS_DIR, f"{model_short_name}_error.log"), 'w') as f:
            f.write(f"Error during training: {str(e)}")
        raise e
    
    # Save model
    trainer.save_model(os.path.join(model_save_dir, "final_model"))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    
    # Calculate F1, precision, recall
    test_metrics = test_results.copy()
    
    # Save results
    results = {
        "model_name": model_name,
        "train_results": {
            "loss": train_result.training_loss,
            "runtime": train_result.metrics["train_runtime"],
            "samples_per_second": train_result.metrics["train_samples_per_second"],
        },
        "validation_results": trainer.state.log_history[-1],  # Last epoch results
        "test_results": test_metrics,
        "hyperparameters": {
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "weight_decay": training_args.weight_decay,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results as JSON
    model_short_name = model_name.split('/')[-1]
    with open(os.path.join(RESULTS_DIR, f"{model_short_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    return {
        "model": model_short_name,
        "test_accuracy": test_metrics["eval_accuracy"],
        "validation_accuracy": trainer.state.log_history[-1]["eval_accuracy"]
    }

def main():
    # Load and prepare data
    df = load_data()
    datasets = prepare_datasets(df)
    
    # Train and evaluate models
    results_summary = []
    
    for model_name in models_to_train:
        try:
            result = train_evaluate_model(model_name, datasets)
            results_summary.append(result)
        except Exception as e:
            print(f"Error training model {model_name}: {str(e)}")
    
    # Print summary of results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Model':<25} {'Test Accuracy':<15} {'Val Accuracy':<15}")
    print("-" * 50)
    for result in results_summary:
        print(f"{result['model']:<25} {result['test_accuracy']:<15.4f} {result['validation_accuracy']:<15.4f}")
    
    # Save overall results
    with open(os.path.join(RESULTS_DIR, "llm_overall_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\nAll models trained and evaluated. Results saved in {MODEL_DIR} and {RESULTS_DIR} directories.")

if __name__ == "__main__":
    # Set environment variable to avoid tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()