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

def generate_embeddings(include_novelty_data=True):
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
    
    # 2. Load novelty scores if available and specified
    if include_novelty_data:
        try:
            print("Loading novelty scores...")
            novelty_df = pd.read_csv('edv_tek_diffusion_patent_novelty_scores.csv')
            print(f"Loaded novelty data for {len(novelty_df)} patents")
            
            # Convert both appln_id columns to string for consistent merging
            print(f"Before conversion - Base df appln_id dtype: {base_df['appln_id'].dtype}")
            print(f"Before conversion - Novelty df appln_id dtype: {novelty_df['appln_id'].dtype}")
            
            # Convert both to string to ensure consistent data types
            base_df['appln_id'] = base_df['appln_id'].astype(str)
            novelty_df['appln_id'] = novelty_df['appln_id'].astype(str)
            
            print(f"After conversion - Base df appln_id dtype: {base_df['appln_id'].dtype}")
            print(f"After conversion - Novelty df appln_id dtype: {novelty_df['appln_id'].dtype}")
            
            # Check for overlapping IDs before merge
            base_ids = set(base_df['appln_id'])
            novelty_ids = set(novelty_df['appln_id'])
            overlap = base_ids.intersection(novelty_ids)
            print(f"Base df unique IDs: {len(base_ids)}")
            print(f"Novelty df unique IDs: {len(novelty_ids)}")
            print(f"Number of overlapping IDs: {len(overlap)}")
            if len(overlap) > 0:
                print(f"Sample overlapping IDs: {list(overlap)[:10]}")
            else:
                print("WARNING: No overlapping application IDs found between datasets!")
            
            # Only keep patents that have novelty scores - use inner join instead of left join
            print("Filtering to only patents with novelty scores...")
            print(f"Patents before filtering: {len(base_df)}")
            base_df = pd.merge(
                base_df, 
                novelty_df[['appln_id', 'novelty_q100', 'novelty_q90', 'novelty_q50']], 
                left_on='appln_id', 
                right_on='appln_id', 
                how='inner'  # Only keep patents that exist in both datasets
            )
            print(f"Patents after filtering (with novelty scores): {len(base_df)}")
            
            # Since we used inner join, all patents should have novelty scores
            print("Calculating novelty percentiles...")
            print(f"Total patents (all should have novelty scores): {len(base_df)}")
            print(f"Patents with novelty scores: {base_df['novelty_q100'].notna().sum()}")
            print(f"Patents without novelty scores: {base_df['novelty_q100'].isna().sum()}")
            
            # Check if we have any patents after filtering
            if len(base_df) == 0:
                print("ERROR: No patents remaining after filtering for novelty scores!")
                print("This means no patents in the diffusion dataset match the novelty scores dataset.")
                return
            
            # Calculate percentiles among all patents (since all should have novelty scores now)
            novelty_cols = [col for col in base_df.columns if col.startswith('novelty_q')]
            for col in novelty_cols:
                base_df[f'{col}_percentile'] = base_df[col].rank(pct=True, method='average') * 100
            
            # Classify patents into novelty categories based on percentiles
            base_df['novelty_category'] = 'Medium'  # Default
            base_df.loc[base_df['novelty_q100_percentile'] >= 90, 'novelty_category'] = 'High'
            base_df.loc[base_df['novelty_q100_percentile'] <= 10, 'novelty_category'] = 'Low'
            
            print(f"Novelty categories: {base_df['novelty_category'].value_counts().to_dict()}")
            
            # Additional debugging information
            print(f"Novelty score statistics:")
            print(f"  Min percentile: {base_df['novelty_q100_percentile'].min():.2f}")
            print(f"  Max percentile: {base_df['novelty_q100_percentile'].max():.2f}")
            print(f"  Mean percentile: {base_df['novelty_q100_percentile'].mean():.2f}")
            print(f"  Patents >= 90th percentile (High): {(base_df['novelty_q100_percentile'] >= 90).sum()}")
            print(f"  Patents <= 10th percentile (Low): {(base_df['novelty_q100_percentile'] <= 10).sum()}")
            print(f"  Patents 11-89th percentile (Medium): {((base_df['novelty_q100_percentile'] > 10) & (base_df['novelty_q100_percentile'] < 90)).sum()}")
        except Exception as e:
            print(f"Warning: Could not load or process novelty scores: {str(e)}")
            print("Continuing without novelty data")
    
    # Save the preprocessed dataframe for the analysis script
    base_df.to_parquet(f'edv_tek_cleantech_patstat_diffusion_preprocessed_novelty_{MIN_YEAR}_{MAX_YEAR}.parquet', index=False)
    print(f"Found {len(base_df)} patents with Y02 classifications")
    
    # 3. Generate embeddings with GPU
    embeddings_file = f'edv_tek_cleantech_patstat_diffusion_embeddings_novelty_{MIN_YEAR}_{MAX_YEAR}.npy'
    
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
    print(f"Preprocessed data saved to edv_tek_cleantech_patstat_diffusion_preprocessed_novelty_{MIN_YEAR}_{MAX_YEAR}.parquet")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate embeddings for patent texts')
    parser.add_argument('--force', action='store_true', help='Force regeneration of embeddings')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                        help='Batch size for GPU processing (default: 64)')
    parser.add_argument('--skip-novelty', action='store_true', help='Skip loading and processing novelty data')
    args = parser.parse_args()
    
    if args.force and os.path.exists(f'edv_tek_cleantech_patstat_diffusion_embeddings_novelty_{MIN_YEAR}_{MAX_YEAR}.npy'):
        os.remove(f'edv_tek_cleantech_patstat_diffusion_embeddings_novelty_{MIN_YEAR}_{MAX_YEAR}.npy')
        print("Removed existing embeddings file, will regenerate")
    
    # Update batch size if specified
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
        print(f"Using custom batch size: {BATCH_SIZE}")
    
    generate_embeddings(include_novelty_data=not args.skip_novelty)