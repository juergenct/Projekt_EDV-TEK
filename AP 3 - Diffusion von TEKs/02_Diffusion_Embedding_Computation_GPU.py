import pandas as pd
import numpy as np
import torch
import ast
import re
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

# Set constants
CPC_Y02_PATTERN = r'Y02[A-Z]'
MIN_YEAR = 1980
MAX_YEAR = 2023
BATCH_SIZE = 64

# Helper functions
def parse_list_string(s):
    """Convert string representation of lists to actual lists."""
    if pd.isna(s) or s == 'None':
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def extract_y02_classes(cpc_classes):
    """Extract Y02 subclasses from CPC classes list."""
    # Handle missing values or empty lists
    if isinstance(cpc_classes, float) and pd.isna(cpc_classes):
        return []
    
    if not isinstance(cpc_classes, str):
        classes = cpc_classes  # Use directly if already a list
    else:
        # Parse if it's a string representation
        classes = parse_list_string(cpc_classes)
    
    # Extract Y02 classes
    y02_classes = []
    for cls in classes:
        if isinstance(cls, str):
            match = re.search(CPC_Y02_PATTERN, cls)
            if match:
                y02_classes.append(match.group(0))
    
    return list(set(y02_classes))  # Return unique Y02 classes

def prepare_text_for_embedding(row):
    """Prepare text for embedding, using only claim text."""
    claims = str(row['claim_fulltext']) if not pd.isna(row['claim_fulltext']) else ""
    
    # Clean and return the claims text
    return claims.strip()

def generate_embeddings():
    """Main function to generate embeddings for patent texts using a single GPU."""
    # 1. Load and preprocess the data
    print("Loading and preprocessing data...")
    df = pd.read_parquet('edv_tek_cleantech_patstat_diffusion.parquet')
    
    # Convert publication date to datetime
    df['publn_date'] = pd.to_datetime(df['publn_date'])
    
    # Filter for patents in the desired year range
    df = df[(df['publn_date'].dt.year >= MIN_YEAR) & (df['publn_date'].dt.year <= MAX_YEAR)]
    
    # Extract Y02 subclasses - this gives us a list for each patent
    df['y02_classes'] = df['cpc_class_symbol'].apply(extract_y02_classes)
    
    # Keep only patents with Y02 classifications
    base_df = df[df['y02_classes'].apply(len) > 0].reset_index(drop=True)
    
    # Save the preprocessed dataframe for the analysis script
    base_df.to_parquet(f'edv_tek_cleantech_patstat_diffusion_preprocessed_{MIN_YEAR}_{MAX_YEAR}.parquet', index=False)
    print(f"Found {len(base_df)} patents with Y02 classifications")
    
    # 2. Generate embeddings with GPU
    embeddings_file = f'edv_tek_cleantech_patstat_diffusion_embeddings_{MIN_YEAR}_{MAX_YEAR}.npy'
    
    if os.path.exists(embeddings_file):
        print(f"Embeddings file {embeddings_file} already exists.")
        print("Use --force flag to regenerate embeddings.")
        return
    
    print("Generating embeddings...")
    
    # Verify GPU is available
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! This script is optimized for GPU usage.")
        print("Processing will be extremely slow on CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Print GPU memory info
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    
    # Load model directly to GPU
    model_name = 'climatebert/distilroberta-base-climate-f'
    model = SentenceTransformer(model_name, device=str(device))
    
    # Process all patents in batches
    all_embeddings = []
    
    # Prepare all text inputs first
    print("Preparing text inputs...")
    texts = []
    for _, row in tqdm(base_df.iterrows(), total=len(base_df), desc="Processing patents"):
        texts.append(prepare_text_for_embedding(row))
    
    # Generate embeddings in batches
    print(f"Generating embeddings in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch_texts = texts[i:i+BATCH_SIZE]
        
        with torch.no_grad():
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        
        # Periodically free GPU memory
        if device.type == 'cuda' and i % (BATCH_SIZE * 10) == 0:
            torch.cuda.empty_cache()
    
    # Combine results
    print("Combining embedding batches...")
    embeddings = np.vstack(all_embeddings)
    
    # Save embeddings to avoid recomputing
    np.save(embeddings_file, embeddings)
    print(f"Generated embeddings with shape {embeddings.shape}")
    print(f"Embeddings saved to {embeddings_file}")
    print(f"Preprocessed data saved to edv_tek_cleantech_patstat_diffusion_preprocessed_{MIN_YEAR}_{MAX_YEAR}.parquet")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate embeddings for patent texts')
    parser.add_argument('--force', action='store_true', help='Force regeneration of embeddings')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                        help='Batch size for GPU processing (default: 64)')
    args = parser.parse_args()
    
    if args.force and os.path.exists('edv_tek_cleantech_patstat_diffusion_embeddings.npy'):
        os.remove('edv_tek_cleantech_patstat_diffusion_embeddings.npy')
        print("Removed existing embeddings file, will regenerate")
    
    # Update batch size if specified
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
        print(f"Using custom batch size: {BATCH_SIZE}")
    
    generate_embeddings()