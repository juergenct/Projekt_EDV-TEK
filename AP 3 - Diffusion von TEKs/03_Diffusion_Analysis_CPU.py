import pandas as pd
import numpy as np
import torch
import ast
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from datetime import datetime
import os
import re
import gc
import faiss
import multiprocessing as mp
from functools import partial
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Set constants
MIN_SIMILARITY_THRESHOLD = 0.92  # Very high threshold for similarity
MAX_SIMILARITY_THRESHOLD = 0.98  # Very high threshold for similarity
# MAX_TIME_LAG_DAYS = 365 * 10  # 10 years in days
MAX_TIME_LAG_DAYS = 365 * 20 # 20 years in days
BATCH_YEARS = 5  # Process in 5-year batches
TOP_K_NEIGHBORS = 20  # Only consider top 20 similar patents
MIN_YEAR = 1980
MAX_YEAR = 2023

# Helper functions
def ensure_scalar(value):
    """Convert NumPy arrays to scalar values."""
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value

def parse_list_string(s):
    """Convert string representation of lists to actual lists."""
    # Handle array inputs directly
    if isinstance(s, (list, np.ndarray)):
        return s.tolist() if isinstance(s, np.ndarray) else s
    
    # Handle None or NaN
    if isinstance(s, float) and pd.isna(s) or s == 'None':
        return []
    
    try:
        result = ast.literal_eval(s)
        # Ensure we return a standard Python list, not a NumPy array
        if isinstance(result, np.ndarray):
            return result.tolist()
        if not isinstance(result, (list, tuple)):
            return [result]
        return list(result)
    except (ValueError, SyntaxError):
        return []

# Function to process a single Y02 class for within-class diffusion
def process_y02_class(args):
    """Process diffusion within a single Y02 class."""
    y02_class, class_df, embeddings = args

    # Print class information
    print(f"Processing class {y02_class} with {len(class_df)} patents")

    # Ensure application IDs are scalar values
    class_df = class_df.copy()

    # Sample 100000 patents if class is too large
    if len(class_df) > 100000:
        print(f"Class {y02_class} is too large, sampling 100000 patents")
        class_df = class_df.sample(100000, random_state=42)

    if isinstance(class_df['appln_id'].iloc[0], np.ndarray):
        class_df['appln_id'] = class_df['appln_id'].apply(lambda x: x.item() if isinstance(x, np.ndarray) else x)
    
    results = []
    
    # Skip classes with too few patents
    if len(class_df) < 2:
        return results
        
    # Sort by publication date
    class_df = class_df.sort_values('publn_date')
    
    # Create fast lookup structures
    class_appln_ids = set(class_df['appln_id'])
    appln_to_date = dict(zip(class_df['appln_id'], class_df['publn_date']))
    appln_to_auth = dict(zip(class_df['appln_id'], class_df['publn_auth']))
    
    # Phase 1: Process citation links using inverted logic
    citation_pairs = set()  # Store only actual citation pairs
    
    # Group by appln_id to process each patent once
    grouped = class_df.groupby('appln_id')
    
    for citing_patent, group in grouped:
        citing_date = appln_to_date[citing_patent]
        citing_auth = appln_to_auth[citing_patent]
        
        # Get all cited patents for this citing patent
        for _, row in group.iterrows():
            if not (isinstance(row['cited_appln_id'], float) and pd.isna(row['cited_appln_id'])) and not (isinstance(row['cited_appln_id'], (list, np.ndarray)) and len(row['cited_appln_id']) == 0):
                cited_ids = parse_list_string(row['cited_appln_id'])
                
                for cited_patent in cited_ids:
                    # Check if cited patent is also in the same Y02 class
                    if cited_patent in class_appln_ids:
                        cited_date = appln_to_date[cited_patent]
                        cited_auth = appln_to_auth[cited_patent]
                        
                        # Check time ordering and lag
                        time_diff = (pd.to_datetime(citing_date) - pd.to_datetime(cited_date)).days
                        if time_diff <= 0 or time_diff > MAX_TIME_LAG_DAYS:
                            continue
                        
                        # Create a unique pair identifier (sorted to avoid duplicates)
                        pair_id = tuple(sorted([cited_patent, citing_patent]))
                        
                        if pair_id not in citation_pairs:
                            citation_pairs.add(pair_id)
                            
                            results.append({
                                'source_id': cited_patent,
                                'target_id': citing_patent,
                                'source_auth': cited_auth,
                                'target_auth': citing_auth,
                                'source_class': y02_class,
                                'target_class': y02_class,
                                'time_lag_days': time_diff,
                                'has_citation': True,
                                'similarity_score': None,  # Don't compute for citations
                                'diffusion_type': 'within',
                                'source_date': cited_date,
                                'target_date': citing_date
                            })
    
    # Phase 2: Process similarity links (for non-citation pairs)
    # Create FAISS index for this class
    original_indices = class_df['original_index'].values
    class_embeddings = embeddings[original_indices]
    
    # Create FAISS index
    dim = class_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(class_embeddings)
    index.add(class_embeddings)
        
    # For each patent, find similar ones respecting time ordering
    for i, (_, source) in enumerate(class_df.iterrows()):
        # Skip patents with no remaining time
        later_patents = class_df[
            (pd.to_datetime(class_df['publn_date']) > pd.to_datetime(source['publn_date'])) &
            (pd.to_datetime(class_df['publn_date']) <= pd.to_datetime(source['publn_date']) + 
             pd.Timedelta(days=MAX_TIME_LAG_DAYS))
        ]
        
        if len(later_patents) == 0:
            continue
        
        # Get source embedding
        source_embedding = embeddings[source['original_index']].reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(source_embedding)
        
        # Search for similar patents
        D, I = index.search(source_embedding, TOP_K_NEIGHBORS)
        
        # Process results
        for j, (sim_score, sim_idx) in enumerate(zip(D[0], I[0])):
            # Skip if similarity is outside threshold range (already correct)
            if not (MIN_SIMILARITY_THRESHOLD <= sim_score <= MAX_SIMILARITY_THRESHOLD):
                continue
            
            # Get the similar patent 
            if sim_idx >= len(class_df):
                continue
                
            target_idx = class_df.index[sim_idx]
            target = class_df.loc[target_idx]
            
            # Skip if target is the same patent
            if source['appln_id'] == target['appln_id']:
                continue
            
            # Skip if target is earlier than source
            if pd.to_datetime(target['publn_date']) <= pd.to_datetime(source['publn_date']):
                continue
            
            # Skip if time lag exceeds maximum
            time_diff = (pd.to_datetime(target['publn_date']) - 
                        pd.to_datetime(source['publn_date'])).days
            if time_diff > MAX_TIME_LAG_DAYS:
                continue
            
            # Skip if already processed as a citation
            pair_key = tuple(sorted([source['appln_id'], target['appln_id']]))
            if pair_key in citation_pairs:
                continue
            
            # Record similarity-based diffusion using fast lookup dictionaries
            results.append({
                'source_id': source['appln_id'],
                'target_id': target['appln_id'],
                'source_auth': appln_to_auth[source['appln_id']],
                'target_auth': appln_to_auth[target['appln_id']],
                'source_class': y02_class,
                'target_class': y02_class,
                'time_lag_days': time_diff,
                'has_citation': False,
                'similarity_score': float(sim_score),
                'diffusion_type': 'within',
                'source_date': appln_to_date[source['appln_id']],
                'target_date': appln_to_date[target['appln_id']]
            })
    
    # Clear index to free memory
    del index
    gc.collect()
    
    return results

def compute_within_class_diffusion(df, embeddings):
    """
    Compute diffusion metrics for patents within the same Y02 class.
    Uses parallel processing for different Y02 classes.
    """
    # Dynamic CPU core utilization
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_workers = int(slurm_cpus)
    else:
        num_workers = mp.cpu_count()
    
    # Create arguments for each Y02 class
    class_args = []
    for y02_class in df['y02_class'].unique():
        class_df = df[df['y02_class'] == y02_class].copy()
        if len(class_df) < 2:  # Skip classes with too few patents
            continue
        class_args.append((y02_class, class_df, embeddings))
    
    print(f"Processing {len(class_args)} Y02 classes using {num_workers} workers")
    
    # Process classes in parallel
    with mp.Pool(processes=min(len(class_args), num_workers)) as pool:
        results_list = list(tqdm_auto(
            pool.imap(process_y02_class, class_args),
            total=len(class_args),
            desc="Processing Y02 classes"
        ))
    
    # Combine results
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

# Function to process a single time batch for between-class diffusion
def process_between_batch(args):
    """Process a single time batch for between-class diffusion."""
    start_year, end_year, df, embeddings = args
    
    # Calculate comparison end year
    comparison_end = min(end_year + (MAX_TIME_LAG_DAYS // 365), df['year'].max())
    
    results = []
    
    # Get source patents (potential "cited" patents)
    source_patents = df[(df['year'] >= start_year) & (df['year'] < end_year)].copy()
    
    # Get target patents (potential "citing" patents) - those that come after source patents
    target_patents = df[(df['year'] >= end_year) & (df['year'] <= comparison_end)].copy()
    
    # Skip if either set is empty
    if len(source_patents) == 0 or len(target_patents) == 0:
        return results
    
    # Create application ID to index mapping for faster lookup
    source_app_to_idx = {row['appln_id']: i for i, (_, row) in enumerate(source_patents.iterrows())}
    
    # Create mapping from application ID to Y02 classes
    source_app_to_classes = source_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    target_app_to_classes = target_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    
    # Create fast lookup dictionaries for metadata (PERFORMANCE OPTIMIZATION)
    source_app_to_date = dict(zip(source_patents['appln_id'], source_patents['publn_date']))
    source_app_to_auth = dict(zip(source_patents['appln_id'], source_patents['publn_auth']))
    
    # Phase 1: Process citation links first
    processed_citation_pairs = set()
    
    # Process citation links between classes
    for _, target in target_patents.iterrows():
        # Skip if no citations
        if isinstance(target['cited_appln_id'], float) and pd.isna(target['cited_appln_id']) or (isinstance(target['cited_appln_id'], (list, np.ndarray)) and len(target['cited_appln_id']) == 0):
            continue
            
        # Get cited application IDs
        cited_app_ids = parse_list_string(target['cited_appln_id'])
        
        # Find cited applications in our source patents
        for cited_app_id in cited_app_ids:
            # Skip if not in our source patents
            if cited_app_id not in source_app_to_idx:
                continue
            
            # Skip if already processed
            pair_key = (cited_app_id, target['appln_id'])
            if pair_key in processed_citation_pairs:
                continue
            
            processed_citation_pairs.add(pair_key)
            
            # Get source Y02 classes
            source_classes = source_app_to_classes.get(cited_app_id, set())
            
            # Get target Y02 classes
            target_classes = target_app_to_classes.get(target['appln_id'], set())
            
            # Find cross-class diffusions
            for source_class in source_classes:
                for target_class in target_classes:
                    # Skip if same class
                    if source_class == target_class:
                        continue
                    
                    # Calculate time difference using fast lookup
                    source_date = source_app_to_date[cited_app_id]
                    time_diff = (pd.to_datetime(target['publn_date']) - pd.to_datetime(source_date)).days
                    
                    # Skip if time lag exceeds maximum
                    if time_diff > MAX_TIME_LAG_DAYS:
                        continue
                    
                    # Record citation-based diffusion using fast lookup
                    results.append({
                        'source_id': cited_app_id,
                        'target_id': target['appln_id'],
                        'source_auth': source_app_to_auth[cited_app_id],
                        'target_auth': target['publn_auth'],
                        'source_class': source_class,
                        'target_class': target_class,
                        'time_lag_days': time_diff,
                        'has_citation': True,
                        'similarity_score': None,  # Don't compute for citations
                        'diffusion_type': 'between',
                        'source_date': source_date,
                        'target_date': target['publn_date']
                    })
    
    # Phase 2: Process similarity links using FAISS
    # Get unique patents
    unique_source_patents = source_patents.drop_duplicates('appln_id').copy()
    unique_target_patents = target_patents.drop_duplicates('appln_id').copy()
    
    # Get embeddings for source patents
    source_original_indices = unique_source_patents['original_index'].values
    source_embeddings = embeddings[source_original_indices]
    
    # Get embeddings for target patents
    target_original_indices = unique_target_patents['original_index'].values
    target_embeddings = embeddings[target_original_indices]
    
    # Build FAISS index for source patents
    dim = source_embeddings.shape[1]
    source_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(source_embeddings)
    source_index.add(source_embeddings)
    
    # Normalize target embeddings
    faiss.normalize_L2(target_embeddings)
    
    # Create mappings for faster lookup
    source_idx_to_app = {i: row['appln_id'] for i, (_, row) in enumerate(unique_source_patents.iterrows())}
    
    # Create fast lookup dictionaries for unique source patents metadata (PERFORMANCE OPTIMIZATION)
    unique_source_app_to_date = dict(zip(unique_source_patents['appln_id'], unique_source_patents['publn_date']))
    unique_source_app_to_auth = dict(zip(unique_source_patents['appln_id'], unique_source_patents['publn_auth']))
    
    # For each target patent, find similar source patents
    for i, (_, target) in enumerate(unique_target_patents.iterrows()):
        target_app_id = target['appln_id']
        target_embedding = target_embeddings[i].reshape(1, -1)
        
        # Search for similar patents in source
        D, I = source_index.search(target_embedding, TOP_K_NEIGHBORS)
        
        # Process results
        for j, (sim_score, sim_idx) in enumerate(zip(D[0], I[0])):
            # Skip if similarity is outside threshold range (already correct)
            if not (MIN_SIMILARITY_THRESHOLD <= sim_score <= MAX_SIMILARITY_THRESHOLD):
                continue
            
            # Get the similar source patent
            if sim_idx >= len(source_idx_to_app):
                continue
                
            source_app_id = source_idx_to_app[sim_idx]
            
            # Skip if already processed as a citation
            pair_key = (source_app_id, target_app_id)
            if pair_key in processed_citation_pairs:
                continue
            
            # Get source Y02 classes
            source_classes = source_app_to_classes.get(source_app_id, set())
            
            # Get target Y02 classes
            target_classes = target_app_to_classes.get(target_app_id, set())
            
            # Find cross-class diffusions
            for source_class in source_classes:
                for target_class in target_classes:
                    # Skip if same class
                    if source_class == target_class:
                        continue
                    
                    # Calculate time difference using fast lookup (PERFORMANCE OPTIMIZATION)
                    source_date = unique_source_app_to_date[source_app_id]
                    time_diff = (pd.to_datetime(target['publn_date']) - pd.to_datetime(source_date)).days
                    
                    # Skip if time lag exceeds maximum
                    if time_diff > MAX_TIME_LAG_DAYS:
                        continue
                    
                    # Record similarity-based diffusion using fast lookup (PERFORMANCE OPTIMIZATION)
                    results.append({
                        'source_id': source_app_id,
                        'target_id': target_app_id,
                        'source_auth': unique_source_app_to_auth[source_app_id],
                        'target_auth': target['publn_auth'],
                        'source_class': source_class,
                        'target_class': target_class,
                        'time_lag_days': time_diff,
                        'has_citation': False,
                        'similarity_score': float(sim_score),
                        'diffusion_type': 'between',
                        'source_date': source_date,
                        'target_date': target['publn_date']
                    })
    
    # Clear memory
    del source_patents, target_patents, source_index
    gc.collect()
    
    return results

def compute_between_class_diffusion(df, embeddings, batch_years=BATCH_YEARS):
    """
    Compute diffusion metrics between different Y02 classes.
    Uses parallel processing for different time batches.
    """
    # Dynamic CPU core utilization
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_workers = int(slurm_cpus)
    else:
        num_workers = mp.cpu_count()
    
    # Add year column for batching if it doesn't exist
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['publn_date']).dt.year
        
    year_min, year_max = df['year'].min(), df['year'].max()
    
    # Create arguments for each time batch
    batch_args = []
    for start_year in range(year_min, year_max, batch_years):
        end_year = start_year + batch_years
        batch_args.append((start_year, end_year, df, embeddings))
    
    print(f"Processing {len(batch_args)} time batches using {num_workers} workers")
    
    # Process batches in parallel
    with mp.Pool(processes=min(len(batch_args), num_workers)) as pool:
        results_list = list(tqdm_auto(
            pool.imap(process_between_batch, batch_args),
            total=len(batch_args),
            desc="Processing time batches for between-class diffusion"
        ))
    
    # Combine results
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

# Function to process a single time batch for directional international diffusion
def process_international_batch_directional(args):
    """Process a single time batch for directional international diffusion."""
    start_year, end_year, df, embeddings, source_auth, target_auth = args
    
    # Get US and EP classes from the data
    us_classes = df[df['publn_auth'] == 'US']['y02_class'].unique()
    ep_classes = df[df['publn_auth'] == 'EP']['y02_class'].unique()
    
    # Filter US and EP patents
    us_patents = df[df['publn_auth'] == 'US']
    ep_patents = df[df['publn_auth'] == 'EP']
    
    # Calculate comparison end year
    comparison_end = min(end_year + (MAX_TIME_LAG_DAYS // 365), df['year'].max())
    
    results = []
    
    # Get source patents
    source_patents = df[(df['publn_auth'] == source_auth) &
                       (df['year'] >= start_year) &
                       (df['year'] < end_year)].copy()
    
    # Get target patents that come after the source patents
    target_patents = df[(df['publn_auth'] == target_auth) &
                       (df['year'] >= end_year) &
                       (df['year'] <= comparison_end)].copy()
    
    # Skip if either set is empty
    if len(source_patents) == 0 or len(target_patents) == 0:
        return {'citation': [], 'semantic': []}
    
    # Create application ID to index mapping for faster lookup
    source_app_to_idx = {row['appln_id']: i for i, (_, row) in enumerate(source_patents.iterrows())}
    
    # Create mapping from application ID to Y02 classes
    source_app_to_classes = source_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    target_app_to_classes = target_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    
    # Create fast lookup dictionaries for metadata (PERFORMANCE OPTIMIZATION)
    source_app_to_date = dict(zip(source_patents['appln_id'], source_patents['publn_date']))
    source_app_to_auth = dict(zip(source_patents['appln_id'], source_patents['publn_auth']))
    
    # Phase 1: Process citation links first
    processed_citation_pairs = set()
    
    # Process international citation links
    for _, target_patent in target_patents.iterrows():
        # Skip if no citations
        if isinstance(target_patent['cited_appln_id'], float) and pd.isna(target_patent['cited_appln_id']) or (isinstance(target_patent['cited_appln_id'], (list, np.ndarray)) and len(target_patent['cited_appln_id']) == 0):
            continue
            
        # Get cited application IDs
        cited_app_ids = parse_list_string(target_patent['cited_appln_id'])
        
        # Find cited US applications
        for cited_app_id in cited_app_ids:
            # Skip if not in our source patents
            if cited_app_id not in source_app_to_idx:
                continue
            
            # Skip if already processed
            pair_key = (cited_app_id, target_patent['appln_id'])
            if pair_key in processed_citation_pairs:
                continue
            
            processed_citation_pairs.add(pair_key)
            
            # Get source Y02 classes
            source_classes = source_app_to_classes.get(cited_app_id, set())
            
            # Get EP Y02 classes
            target_classes = target_app_to_classes.get(target_patent['appln_id'], set())
            
            # Find all class combinations
            for source_class in source_classes:
                for target_class in target_classes:
                    # Calculate time difference using fast lookup (PERFORMANCE OPTIMIZATION)
                    source_date = source_app_to_date[cited_app_id]
                    time_diff = (pd.to_datetime(target_patent['publn_date']) - pd.to_datetime(source_date)).days
                    
                    # Skip if time lag exceeds maximum
                    if time_diff > MAX_TIME_LAG_DAYS:
                        continue
                    
                    # Record citation-based diffusion
                    results.append({
                        'source_id': cited_app_id,
                        'target_id': target_patent['appln_id'],
                        'source_auth': source_auth,
                        'target_auth': target_auth,
                        'source_class': source_class,
                        'target_class': target_class,
                        'time_lag_days': time_diff,
                        'has_citation': True,
                        'similarity_score': None,  # Don't compute for citations
                        'diffusion_type': 'international',
                        'source_date': source_date,
                        'target_date': target_patent['publn_date']
                    })
    
    # Phase 2: Process similarity links using FAISS
    # Filter to unique patents to avoid duplicate embeddings
    unique_source_patents = source_patents.drop_duplicates('appln_id').copy()
    unique_target_patents = target_patents.drop_duplicates('appln_id').copy()
    
    # Get embeddings for source patents
    source_original_indices = unique_source_patents['original_index'].values
    source_embeddings = embeddings[source_original_indices]
    
    # Get embeddings for target patents
    target_original_indices = unique_target_patents['original_index'].values
    target_embeddings = embeddings[target_original_indices]
    
    # Build FAISS index for source patents
    dim = source_embeddings.shape[1]
    source_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(source_embeddings)
    source_index.add(source_embeddings)
    
    # Normalize target embeddings
    faiss.normalize_L2(target_embeddings)
    
    # Create mapping for faster lookup
    source_idx_to_app = {i: row['appln_id'] for i, (_, row) in enumerate(unique_source_patents.iterrows())}
    
    # Create fast lookup dictionaries for unique source patents metadata (PERFORMANCE OPTIMIZATION)
    unique_source_app_to_date = dict(zip(unique_source_patents['appln_id'], unique_source_patents['publn_date']))
    unique_source_app_to_auth = dict(zip(unique_source_patents['appln_id'], unique_source_patents['publn_auth']))
    
    # For each target patent, find similar source patents
    for i, (_, target_patent) in enumerate(unique_target_patents.iterrows()):
        target_app_id = target_patent['appln_id']
        target_embedding = target_embeddings[i].reshape(1, -1)
        
        # Search for similar source patents
        D, I = source_index.search(target_embedding, TOP_K_NEIGHBORS)
        
        # Process results
        for j, (sim_score, sim_idx) in enumerate(zip(D[0], I[0])):
            # Skip if similarity is outside threshold range (already correct)
            if not (MIN_SIMILARITY_THRESHOLD <= sim_score <= MAX_SIMILARITY_THRESHOLD):
                continue
            
            # Get the similar source patent
            if sim_idx >= len(source_idx_to_app):
                continue
                
            source_app_id = source_idx_to_app[sim_idx]
            
            # Skip if already processed as a citation
            pair_key = (source_app_id, target_app_id)
            if pair_key in processed_citation_pairs:
                continue
            
            # Get US Y02 classes
            source_classes = source_app_to_classes.get(source_app_id, set())
            
            # Get target Y02 classes
            target_classes = target_app_to_classes.get(target_app_id, set())
            
            # Find all class combinations
            for source_class in source_classes:
                for target_class in target_classes:
                    # Calculate time difference using fast lookup (PERFORMANCE OPTIMIZATION)
                    source_date = unique_source_app_to_date[source_app_id]
                    time_diff = (pd.to_datetime(target_patent['publn_date']) - pd.to_datetime(source_date)).days
                    
                    # Skip if time lag exceeds maximum
                    if time_diff > MAX_TIME_LAG_DAYS:
                        continue
                    
                    # Record similarity-based diffusion using fast lookup (PERFORMANCE OPTIMIZATION)
                    results.append({
                        'source_id': source_app_id,
                        'target_id': target_app_id,
                        'source_auth': source_auth,
                        'target_auth': target_auth,
                        'source_class': source_class,
                        'target_class': target_class,
                        'time_lag_days': time_diff,
                        'has_citation': False,
                        'similarity_score': float(sim_score),
                        'diffusion_type': 'international',
                        'source_date': source_date,
                        'target_date': target_patent['publn_date']
                    })
    
    # Clear memory
    del source_patents, target_patents, source_index
    gc.collect()
    
    # Split results into citation and semantic
    citation_results = [r for r in results if r['has_citation']]
    semantic_results = [r for r in results if not r['has_citation']]
    
    return {
        'citation': citation_results,
        'semantic': semantic_results
    }

def compute_international_diffusion(df, embeddings, batch_years=BATCH_YEARS):
    """
    Compute bidirectional international diffusion metrics (US<->EP patents).
    Uses parallel processing for different time batches and directions.
    """
    # Dynamic CPU core utilization
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_workers = int(slurm_cpus)
    else:
        num_workers = mp.cpu_count()
    
    # Add year column for batching if it doesn't exist
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['publn_date']).dt.year
        
    year_min, year_max = df['year'].min(), df['year'].max()
    
    # Create arguments for each time batch
    batch_args = []
    for start_year in range(year_min, year_max, batch_years):
        end_year = start_year + batch_years
        # Add arguments for both directions
        batch_args.append((start_year, end_year, df, embeddings, 'US', 'EP'))
        batch_args.append((start_year, end_year, df, embeddings, 'EP', 'US'))
    
    print(f"Processing {len(batch_args)} time batches using {num_workers} workers")
    
    # Process batches in parallel
    with mp.Pool(processes=min(len(batch_args), num_workers)) as pool:
        results_list = list(tqdm_auto(
            pool.imap(process_international_batch_directional, batch_args),
            total=len(batch_args),
            desc="Processing time batches for bidirectional international diffusion"
        ))
    
    # Combine and separate results
    all_citation_results = []
    all_semantic_results = []
    
    for batch_results in results_list:
        all_citation_results.extend(batch_results['citation'])
        all_semantic_results.extend(batch_results['semantic'])
    
    # Create output directories
    output_dir = os.path.join('results', 'international_diffusion')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'citation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'semantic'), exist_ok=True)
    
    # Save citation results
    if all_citation_results:
        citation_df = pd.DataFrame(all_citation_results)
        for (source, target), group in citation_df.groupby(['source_auth', 'target_auth']):
            citation_file = os.path.join(output_dir, 'citation',
                                       f'international_diffusion_{source}_to_{target}_citation.csv')
            group.to_csv(citation_file, index=False)

    # Save semantic results
    if all_semantic_results:
        semantic_df = pd.DataFrame(all_semantic_results)
        for (source, target), group in semantic_df.groupby(['source_auth', 'target_auth']):
            semantic_file = os.path.join(output_dir, 'semantic',
                                       f'international_diffusion_{source}_to_{target}_semantic.csv')
            group.to_csv(semantic_file, index=False)
    
    # Return combined results for visualization
    return pd.DataFrame(all_citation_results + all_semantic_results)

def create_visualizations(diffusion_df, patents_df):
    """Create visualizations for diffusion analysis and save as HTML and PNG files."""
    # Create base visualization directory and mechanism-specific subdirectories
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('visualizations/citation', exist_ok=True)
    os.makedirs('visualizations/semantic', exist_ok=True)
    os.makedirs('visualizations/combined', exist_ok=True)
    
    # Create visualizations for each mechanism separately and combined
    for mechanism, suffix, title_suffix in [
        (diffusion_df[diffusion_df['has_citation']], '_citation', ' (Citation-based)'),
        (diffusion_df[~diffusion_df['has_citation']], '_semantic', ' (Semantic Similarity-based)'),
        (diffusion_df, '', '')
    ]:
        if len(mechanism) == 0:
            continue
            
        # Determine output directory
        output_dir = 'visualizations/combined' if not suffix else f'visualizations/{suffix[1:]}'
        
        # 1. Diffusion time distribution by type
        fig = go.Figure()
        
        for diff_type in mechanism['diffusion_type'].unique():
            subset = mechanism[mechanism['diffusion_type'] == diff_type]
            fig.add_trace(go.Histogram(
                x=subset['time_lag_days'] / 365.25,
                name=diff_type,
                opacity=0.7,
                nbinsx=20
            ))
        
        fig.update_layout(
            title=f'Distribution of Diffusion Time by Diffusion Type{title_suffix}',
            xaxis_title='Diffusion Time (Years)',
            yaxis_title='Count',
            barmode='overlay',
            legend_title='Diffusion Type'
        )
        
        # Save both HTML and PNG versions
        fig.write_html(f'{output_dir}/diffusion_time_distribution{suffix}.html')
        fig.write_image(f'{output_dir}/diffusion_time_distribution{suffix}.png', engine='kaleido')
    
    # 2. Diffusion network between Y02 classes
    for mechanism, suffix, title_suffix in [
        (diffusion_df[(diffusion_df['diffusion_type'] == 'between') & (diffusion_df['has_citation'])], '_citation', ' (Citation-based)'),
        (diffusion_df[(diffusion_df['diffusion_type'] == 'between') & (~diffusion_df['has_citation'])], '_semantic', ' (Semantic Similarity-based)'),
        (diffusion_df[diffusion_df['diffusion_type'] == 'between'], '', '')
    ]:
        if len(mechanism) == 0:
            continue
            
        output_dir = 'visualizations/combined' if not suffix else f'visualizations/{suffix[1:]}'
        between_df = mechanism
        G = nx.DiGraph()
        
        # Add edges with weight based on count
        class_counts = between_df.groupby(['source_class', 'target_class']).size().reset_index(name='count')
        for _, row in class_counts.iterrows():
            G.add_edge(row['source_class'], row['target_class'], weight=row['count'])
        
        # Calculate node sizes based on degree
        node_sizes = {node: G.degree(node) * 10 for node in G.nodes()}
        
        # Create positions for nodes
        pos = nx.spring_layout(G, seed=42)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            # Create arrow shape
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=weight/max(class_counts['count'])*10, color='#888'),
                hoverinfo='text',
                text=f'{edge[0]} → {edge[1]}: {weight} connections',
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=[f"{node}: {G.degree(node)} connections" for node in G.nodes()],
            mode='markers+text',
            textposition='top center',
            marker=dict(
                size=[node_sizes[node] for node in G.nodes()],
                color='skyblue',
                line=dict(width=2, color='#333')
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=f'Diffusion Network Between Y02 Classes{title_suffix}',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800
        )
        
        fig.write_html(f'{output_dir}/y02_diffusion_network{suffix}.html')
        fig.write_image(f'{output_dir}/y02_diffusion_network{suffix}.png', engine='kaleido')
    
    # 3. International diffusion heatmap
    for mechanism, suffix, title_suffix in [
        (diffusion_df[(diffusion_df['diffusion_type'] == 'international') & (diffusion_df['has_citation'])], '_citation', ' (Citation-based)'),
        (diffusion_df[(diffusion_df['diffusion_type'] == 'international') & (~diffusion_df['has_citation'])], '_semantic', ' (Semantic Similarity-based)'),
        (diffusion_df[diffusion_df['diffusion_type'] == 'international'], '', '')
    ]:
        if len(mechanism) == 0:
            continue
            
        output_dir = 'visualizations/combined' if not suffix else f'visualizations/{suffix[1:]}'
        int_df = mechanism
        
        if not int_df.empty:
            # Create separate heatmaps for each direction
            directions = int_df[['source_auth', 'target_auth']].drop_duplicates().values
            
            for source_auth, target_auth in directions:
                dir_df = int_df[(int_df['source_auth'] == source_auth) &
                               (int_df['target_auth'] == target_auth)]
                
                dir_stats = dir_df.groupby(['source_class', 'target_class'])['time_lag_days'].mean().reset_index()
                
                # Create a pivot table for the heatmap
                heatmap_data = dir_stats.pivot(index='source_class', columns='target_class', values='time_lag_days') / 365.25
                
                # Convert to format needed for plotly
                z_data = heatmap_data.values
                x_data = heatmap_data.columns.tolist()
                y_data = heatmap_data.index.tolist()
                
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=z_data,
                    x=x_data,
                    y=y_data,
                    colorscale='YlGnBu',
                    hoverongaps=False,
                    text=[[f"{z:.1f} years" for z in row] for row in z_data],
                    hoverinfo='text+x+y'
                ))
                
                heatmap_fig.update_layout(
                    title=f'Average Diffusion Time (Years) from {source_auth} to {target_auth} Patents by Y02 Class{title_suffix}',
                    xaxis_title='Target Y02 Class',
                    yaxis_title='Source Y02 Class'
                )
                
                # Save with direction and mechanism suffix
                direction_suffix = f'_{source_auth}_to_{target_auth}{suffix}'
                heatmap_fig.write_html(f'{output_dir}/international_diffusion_heatmap{direction_suffix}.html')
                heatmap_fig.write_image(f'{output_dir}/international_diffusion_heatmap{direction_suffix}.png', engine='kaleido')
    
    # 4. Time series of diffusion over years
    for mechanism, suffix, title_suffix in [
        (diffusion_df[diffusion_df['has_citation']], '_citation', ' (Citation-based)'),
        (diffusion_df[~diffusion_df['has_citation']], '_semantic', ' (Semantic Similarity-based)'),
        (diffusion_df, '', '')
    ]:
        if len(mechanism) == 0:
            continue
            
        output_dir = 'visualizations/combined' if not suffix else f'visualizations/{suffix[1:]}'
        mechanism['year'] = pd.to_datetime(mechanism['target_date']).dt.year
        time_series = mechanism.groupby(['year', 'diffusion_type']).size().reset_index(name='count')
        
        fig = go.Figure()
        
        for diff_type in time_series['diffusion_type'].unique():
            data = time_series[time_series['diffusion_type'] == diff_type]
            fig.add_trace(go.Scatter(
                x=data['year'],
                y=data['count'],
                mode='lines+markers',
                name=diff_type,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f'TEK Diffusion Over Time by Type{title_suffix}',
            xaxis_title='Year',
            yaxis_title='Number of Diffusion Events',
            hovermode='x unified',
            legend_title='Diffusion Type'
        )
        
        fig.write_html(f'{output_dir}/diffusion_time_series{suffix}.html')
        fig.write_image(f'{output_dir}/diffusion_time_series{suffix}.png', engine='kaleido')
    
    # 5. Add diffusion speed comparison chart
    for mechanism, suffix, title_suffix in [
        (diffusion_df[diffusion_df['has_citation']], '_citation', ' (Citation-based)'),
        (diffusion_df[~diffusion_df['has_citation']], '_semantic', ' (Semantic Similarity-based)'),
        (diffusion_df, '', '')
    ]:
        if len(mechanism) == 0:
            continue
            
        output_dir = 'visualizations/combined' if not suffix else f'visualizations/{suffix[1:]}'
        class_speed = mechanism.groupby(['source_class', 'diffusion_type'])['time_lag_days'].mean().reset_index()
        class_speed['time_lag_years'] = class_speed['time_lag_days'] / 365.25
        
        speed_fig = px.bar(
            class_speed,
            x='source_class',
            y='time_lag_years',
            color='diffusion_type',
            barmode='group',
            title=f'Average Diffusion Speed by Y02 Class and Diffusion Type{title_suffix}',
            labels={'time_lag_years': 'Average Diffusion Time (Years)', 'source_class': 'Y02 Class'},
            text_auto='.1f'
        )
        
        speed_fig.update_layout(
            legend_title='Diffusion Type',
            xaxis_tickangle=-45
        )
        
        speed_fig.write_html(f'{output_dir}/diffusion_speed_comparison{suffix}.html')
        speed_fig.write_image(f'{output_dir}/diffusion_speed_comparison{suffix}.png', engine='kaleido')
    
    # 6. Combined mechanism comparison (only in combined directory)
    diffusion_df['diffusion_mechanism'] = diffusion_df['has_citation'].map({True: 'Citation', False: 'Semantic Similarity'})
    mechanism_counts = diffusion_df.groupby(['diffusion_type', 'diffusion_mechanism']).size().reset_index(name='count')
    
    mech_fig = px.bar(
        mechanism_counts,
        x='diffusion_type',
        y='count',
        color='diffusion_mechanism',
        title='Diffusion Mechanisms by Type',
        labels={'diffusion_type': 'Diffusion Type', 'count': 'Number of Connections'},
        barmode='group'
    )
    
    mech_fig.write_html('visualizations/combined/diffusion_mechanisms.html')
    mech_fig.write_image('visualizations/combined/diffusion_mechanisms.png', engine='kaleido')
    
    print("Interactive HTML visualizations saved to 'visualizations' directory with separate citation and semantic subdirectories.")

def analyze_diffusion():
    """Main function to analyze diffusion patterns using pre-computed embeddings with multiprocessing."""
    # Get start time for performance tracking
    start_time = datetime.now()
    
    # Check available CPU resources
    num_cores = mp.cpu_count()
    print(f"Running on a machine with {num_cores} CPU cores")
    
    # Check if required files exist
    if not os.path.exists(f'/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_cleantech_patstat_diffusion_preprocessed_{MIN_YEAR}_{MAX_YEAR}.parquet'):
        print(f"Error: preprocessed data file not found.")
        print("Please run generate_embeddings.py first.")
        return
    
    if not os.path.exists(f'/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_cleantech_patstat_diffusion_embeddings_{MIN_YEAR}_{MAX_YEAR}.npy'):
        print("Error: embeddings file not found.")
        print("Please run generate_embeddings.py first.")
        return
    
    # Load preprocessed data and embeddings
    print("Loading preprocessed data and embeddings...")
    base_df = pd.read_parquet(f'/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_cleantech_patstat_diffusion_preprocessed_{MIN_YEAR}_{MAX_YEAR}.parquet')
    embeddings = np.load(f'/fibus/fs1/0f/cyh1826/wt/edv_tek/edv_tek_cleantech_patstat_diffusion_embeddings_{MIN_YEAR}_{MAX_YEAR}.npy')
    
    # Verify alignment
    print(f"Base dataframe size: {len(base_df)}, Embeddings size: {len(embeddings)}")
    assert len(base_df) == len(embeddings), "Dataframe and embeddings must have the same length"
    base_df['embedding_idx'] = np.arange(len(base_df))

    # Explode the dataframe by Y02 classes
    print("Exploding dataframe by Y02 classes...")
    # Check the type of y02_classes and handle appropriately
    first_item = base_df['y02_classes'].iloc[0]
    if isinstance(first_item, np.ndarray):
        print(f"Note: y02_classes contains NumPy arrays, which is compatible with explode()")
    elif not isinstance(first_item, (list, tuple)):
        print(f"Warning: y02_classes contains type {type(first_item)}, which may cause issues")
        # Try to convert to list if possible
        try:
            print("Attempting to convert y02_classes to lists...")
            base_df['y02_classes'] = base_df['y02_classes'].apply(lambda x: list(x) if hasattr(x, '__iter__') else [x])
            print("Conversion successful")
        except Exception as e:
            print(f"Error converting to lists: {e}")

    # Handle empty classes
    base_df['y02_classes'] = base_df['y02_classes'].apply(lambda x: x if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 else [])

    # Now explode the dataframe
    df = base_df.explode('y02_classes').rename(columns={'y02_classes': 'y02_class'})
    
    # Remove rows with empty or null y02_class
    df = df[~pd.isna(df['y02_class'])].reset_index(drop=True)
    df = df[df['y02_class'] != ''].reset_index(drop=True)

    # Save original index for embedding reference
    df['original_index'] = df['embedding_idx']

    df = df.reset_index(drop=True)

    print(f"Exploded dataframe has {len(df)} patent-class combinations")
    
    # Set up overall progress tracking
    with tqdm_auto(total=3, desc="Overall analysis progress", position=0) as pbar:
        # 1. Compute within-class diffusion
        print("\nComputing within-class diffusion...")
        within_diffusion = compute_within_class_diffusion(df, embeddings)
        within_diffusion.to_csv('within_class_diffusion.csv', index=False)
        print(f"Found {len(within_diffusion)} within-class diffusion connections")
        pbar.update(1)
        
        # 2. Compute between-class diffusion
        print("\nComputing between-class diffusion...")
        between_diffusion = compute_between_class_diffusion(df, embeddings)
        between_diffusion.to_csv('between_class_diffusion.csv', index=False)
        print(f"Found {len(between_diffusion)} between-class diffusion connections")
        pbar.update(1)
        
        # 3. Compute international diffusion
        print("\nComputing international diffusion...")
        international_diffusion = compute_international_diffusion(df, embeddings)
        international_diffusion.to_csv('international_diffusion.csv', index=False)
        print(f"Found {len(international_diffusion)} international diffusion connections")
        pbar.update(1)
    
    # 4. Compute summary statistics
    all_diffusion = pd.concat([within_diffusion, between_diffusion, international_diffusion])
    
    # Average diffusion time by type
    diffusion_time_stats = all_diffusion.groupby('diffusion_type')['time_lag_days'].agg(['mean', 'median', 'count'])
    print("\nAverage diffusion time by type:")
    print(diffusion_time_stats)
    
    # Average diffusion time by Y02 class
    class_diffusion_stats = all_diffusion.groupby(['source_class', 'target_class'])['time_lag_days'].agg(['mean', 'median', 'count'])
    class_diffusion_stats = class_diffusion_stats.sort_values('count', ascending=False)
    
    print("\nTop class pairs by diffusion count:")
    print(class_diffusion_stats.head(10))
    
    # Diffusion mechanism distribution
    mechanism_counts = all_diffusion.groupby(['diffusion_type', 'has_citation']).size().unstack()
    print("\nDiffusion mechanism distribution:")
    print(mechanism_counts)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(all_diffusion, df)
    
    # Calculate and display total execution time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nAnalysis complete! Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    analyze_diffusion()