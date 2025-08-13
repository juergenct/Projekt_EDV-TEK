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
from plotly.subplots import make_subplots
from enum import Enum, auto

class AnalysisType(Enum):
    CITATION = auto()
    SEMANTIC = auto()

# Set constants
MIN_SIMILARITY_THRESHOLD = 0.92  # Minimum similarity threshold
MAX_SIMILARITY_THRESHOLD = 0.98  # Maximum similarity threshold
MAX_TIME_LAG_DAYS = 1825  # 5 years maximum time lag
BATCH_YEARS = 5  # Process in 5-year batches
TOP_K_NEIGHBORS = 20  # Only consider top 20 similar patents
MIN_YEAR = 1980
MAX_YEAR = 2023

# Novelty categories to analyze
NOVELTY_CATEGORIES = ['High', 'Medium', 'Low']  # Analyze all three novelty categories

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
    if (isinstance(s, float) and pd.isna(s)) or s == 'None':
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
    y02_class, class_df, embeddings, novelty_category, analysis_type = args

    # Print class information
    print(f"Processing class {y02_class} with {len(class_df)} patents (Novelty: {novelty_category}, Analysis: {analysis_type.name})")

    # Ensure application IDs are scalar values
    class_df = class_df.copy()

    # Sample 5000 patents if class is too large
    if len(class_df) > 5000:
        print(f"Class {y02_class} is too large, sampling 5000 patents")
        class_df = class_df.sample(5000, random_state=42)

    if isinstance(class_df['appln_id'].iloc[0], np.ndarray):
        class_df['appln_id'] = class_df['appln_id'].apply(lambda x: x.item() if isinstance(x, np.ndarray) else x)
    
    results = []
    
    # Skip classes with too few patents
    if len(class_df) < 2:
        return results

    # Determine similarity threshold based on analysis type
    similarity_threshold = (
        MIN_SIMILARITY_THRESHOLD if analysis_type == AnalysisType.SEMANTIC
        else MAX_SIMILARITY_THRESHOLD
    )
        
    # Sort by publication date
    class_df = class_df.sort_values('publn_date')
    
    # Create fast lookup dictionaries for patent metadata
    appln_to_date = {row['appln_id']: row['publn_date'] for _, row in class_df.iterrows()}
    appln_to_auth = {row['appln_id']: row['publn_auth'] for _, row in class_df.iterrows()}
    
    # Phase 1: Process citation links with inverted citation logic
    citation_pairs = set()
    
    # Build inverted citation index for faster lookup
    for _, target in class_df.iterrows():
        if not (isinstance(target['cited_appln_id'], float) and pd.isna(target['cited_appln_id'])) and not (isinstance(target['cited_appln_id'], (list, np.ndarray)) and len(target['cited_appln_id']) == 0):
            cited_ids = parse_list_string(target['cited_appln_id'])
            
            for cited_id in cited_ids:
                # Check if cited patent is in our class
                if cited_id in appln_to_date:
                    # Verify time ordering and lag
                    source_date = appln_to_date[cited_id]
                    target_date = target['publn_date']
                    time_diff = (pd.to_datetime(target_date) - pd.to_datetime(source_date)).days
                    
                    if 0 < time_diff <= MAX_TIME_LAG_DAYS:
                        pair_key = (cited_id, target['appln_id'])
                        citation_pairs.add(pair_key)
                        
                        # Record citation-based diffusion
                        results.append({
                            'source_id': cited_id,
                            'target_id': target['appln_id'],
                            'source_auth': appln_to_auth[cited_id],
                            'target_auth': target['publn_auth'],
                            'source_class': y02_class,
                            'target_class': y02_class,
                            'time_lag_days': time_diff,
                            'has_citation': True,
                            'similarity_score': None,  # Don't compute for citations
                            'diffusion_type': 'within',
                            'source_date': source_date,
                            'target_date': target_date,
                            'novelty_category': novelty_category
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
            # Skip if similarity is below threshold
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
            pair_key = (source['appln_id'], target['appln_id'])
            if pair_key in citation_pairs:
                continue
            
            # Record similarity-based diffusion
            results.append({
                'source_id': source['appln_id'],
                'target_id': target['appln_id'],
                'source_auth': source['publn_auth'],
                'target_auth': target['publn_auth'],
                'source_class': y02_class,
                'target_class': y02_class,
                'time_lag_days': time_diff,
                'has_citation': False,
                'similarity_score': float(sim_score),
                'diffusion_type': 'within',
                'source_date': source['publn_date'],
                'target_date': target['publn_date'],
                'novelty_category': novelty_category
            })
    
    # Clear index to free memory
    del index
    gc.collect()
    
    return results

def compute_within_class_diffusion(df, embeddings, novelty_category, analysis_type=AnalysisType.SEMANTIC):
    """
    Compute diffusion metrics for patents within the same Y02 class.
    Uses parallel processing for different Y02 classes.
    
    Args:
        df: DataFrame of patents
        embeddings: Patent embeddings
        novelty_category: Which novelty category to analyze ('High', 'Medium' or 'Low')
        analysis_type: Type of analysis to perform (CITATION or SEMANTIC)
    """
    # Determine optimal number of workers with SLURM support
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    
    # Create arguments for each Y02 class
    class_args = []
    for y02_class in df['y02_class'].unique():
        class_df = df[df['y02_class'] == y02_class].copy()
        if len(class_df) < 2:  # Skip classes with too few patents
            continue
        class_args.append((y02_class, class_df, embeddings, novelty_category, analysis_type))
    
    print(f"Processing {len(class_args)} Y02 classes for {novelty_category} novelty patents using {num_workers} workers")
    
    # Process classes in parallel
    with mp.Pool(processes=min(len(class_args), num_workers)) as pool:
        results_list = list(tqdm_auto(
            pool.imap(process_y02_class, class_args),
            total=len(class_args),
            desc=f"Processing Y02 classes ({novelty_category} novelty)"
        ))
    
    # Combine results
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

# Function to process a single time batch for between-class diffusion
def process_between_batch(args):
    """Process a single time batch for between-class diffusion."""
    start_year, end_year, df, embeddings, novelty_category = args
    
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
    
    # Create fast lookup dictionaries for source patent metadata
    source_app_to_date = {row['appln_id']: row['publn_date'] for _, row in source_patents.iterrows()}
    source_app_to_auth = {row['appln_id']: row['publn_auth'] for _, row in source_patents.iterrows()}
    
    # Create application ID to index mapping for faster lookup
    source_app_to_idx = {row['appln_id']: i for i, (_, row) in enumerate(source_patents.iterrows())}
    
    # Create mapping from application ID to Y02 classes
    source_app_to_classes = source_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    target_app_to_classes = target_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    
    # Phase 1: Process citation links first
    processed_citation_pairs = set()
    
    # Process citation links between classes
    for _, target in target_patents.iterrows():
        # Skip if no citations
        if (isinstance(target['cited_appln_id'], float) and pd.isna(target['cited_appln_id'])) or (isinstance(target['cited_appln_id'], (list, np.ndarray)) and len(target['cited_appln_id']) == 0):
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
                    
                    # Record citation-based diffusion
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
                        'target_date': target['publn_date'],
                        'novelty_category': novelty_category
                    })
    
    # Phase 2: Process similarity links using FAISS
    # Get unique patents
    unique_source_patents = source_patents.drop_duplicates('appln_id').copy()
    unique_target_patents = target_patents.drop_duplicates('appln_id').copy()
    
    # Create fast lookup dictionaries for unique patents
    unique_source_app_to_date = {row['appln_id']: row['publn_date'] for _, row in unique_source_patents.iterrows()}
    unique_source_app_to_auth = {row['appln_id']: row['publn_auth'] for _, row in unique_source_patents.iterrows()}
    
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
    
    # For each target patent, find similar source patents
    for i, (_, target) in enumerate(unique_target_patents.iterrows()):
        target_app_id = target['appln_id']
        target_embedding = target_embeddings[i].reshape(1, -1)
        
        # Search for similar patents in source
        D, I = source_index.search(target_embedding, TOP_K_NEIGHBORS)
        
        # Process results
        for j, (sim_score, sim_idx) in enumerate(zip(D[0], I[0])):
            # Skip if similarity is below threshold
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
                    
                    # Calculate time difference using fast lookup
                    source_date = unique_source_app_to_date[source_app_id]
                    time_diff = (pd.to_datetime(target['publn_date']) - pd.to_datetime(source_date)).days
                    
                    # Skip if time lag exceeds maximum
                    if time_diff > MAX_TIME_LAG_DAYS:
                        continue
                    
                    # Record similarity-based diffusion
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
                        'target_date': target['publn_date'],
                        'novelty_category': novelty_category
                    })
    
    # Clear memory
    del source_patents, target_patents, source_index
    gc.collect()
    
    return results

def compute_between_class_diffusion(df, embeddings, novelty_category, batch_years=BATCH_YEARS):
    """
    Compute diffusion metrics between different Y02 classes.
    Uses parallel processing for different time batches.
    
    Args:
        df: DataFrame of patents
        embeddings: Patent embeddings
        novelty_category: Which novelty category to analyze ('High' or 'Low')
        batch_years: Number of years to process in each batch
    """
    # Determine optimal number of workers with SLURM support
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    
    # Add year column for batching if it doesn't exist
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['publn_date']).dt.year
        
    year_min, year_max = df['year'].min(), df['year'].max()
    
    # Create arguments for each time batch
    batch_args = []
    for start_year in range(year_min, year_max, batch_years):
        end_year = start_year + batch_years
        batch_args.append((start_year, end_year, df, embeddings, novelty_category))
    
    print(f"Processing {len(batch_args)} time batches for {novelty_category} novelty patents using {num_workers} workers")
    
    # Process batches in parallel
    with mp.Pool(processes=min(len(batch_args), num_workers)) as pool:
        results_list = list(tqdm_auto(
            pool.imap(process_between_batch, batch_args),
            total=len(batch_args),
            desc=f"Processing time batches for between-class diffusion ({novelty_category} novelty)"
        ))
    
    # Combine results
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

# Function to process a single time batch for directional international diffusion
def process_international_batch_directional(args):
    """Process a single time batch for directional international diffusion."""
    start_year, end_year, df, embeddings, novelty_category, source_auth, target_auth = args
    
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
    
    # Create fast lookup dictionaries for source patent metadata
    source_app_to_date = {row['appln_id']: row['publn_date'] for _, row in source_patents.iterrows()}
    
    # Create application ID to index mapping for faster lookup
    source_app_to_idx = {row['appln_id']: i for i, (_, row) in enumerate(source_patents.iterrows())}
    
    # Create mapping from application ID to Y02 classes
    source_app_to_classes = source_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    target_app_to_classes = target_patents.groupby('appln_id')['y02_class'].apply(set).to_dict()
    
    # Phase 1: Process citation links first
    processed_citation_pairs = set()
    
    # Process international citation links
    for _, target_patent in target_patents.iterrows():
        # Skip if no citations
        if (isinstance(target_patent['cited_appln_id'], float) and pd.isna(target_patent['cited_appln_id'])) or (isinstance(target_patent['cited_appln_id'], (list, np.ndarray)) and len(target_patent['cited_appln_id']) == 0):
            continue
            
        # Get cited application IDs
        cited_app_ids = parse_list_string(target_patent['cited_appln_id'])
        
        # Find cited source applications
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
            
            # Get target Y02 classes
            target_classes = target_app_to_classes.get(target_patent['appln_id'], set())
            
            # Find all class combinations
            for source_class in source_classes:
                for target_class in target_classes:
                    # Calculate time difference using fast lookup
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
                        'target_date': target_patent['publn_date'],
                        'novelty_category': novelty_category
                    })
    
    # Phase 2: Process similarity links using FAISS
    # Filter to unique patents to avoid duplicate embeddings
    unique_source_patents = source_patents.drop_duplicates('appln_id').copy()
    unique_target_patents = target_patents.drop_duplicates('appln_id').copy()
    
    # Create fast lookup dictionaries for unique patents
    unique_source_app_to_date = {row['appln_id']: row['publn_date'] for _, row in unique_source_patents.iterrows()}
    
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
    
    # For each target patent, find similar source patents
    for i, (_, target_patent) in enumerate(unique_target_patents.iterrows()):
        target_app_id = target_patent['appln_id']
        target_embedding = target_embeddings[i].reshape(1, -1)
        
        # Search for similar source patents
        D, I = source_index.search(target_embedding, TOP_K_NEIGHBORS)
        
        # Process results
        for j, (sim_score, sim_idx) in enumerate(zip(D[0], I[0])):
            # Skip if similarity is below threshold
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
            
            # Find all class combinations
            for source_class in source_classes:
                for target_class in target_classes:
                    # Calculate time difference using fast lookup
                    source_date = unique_source_app_to_date[source_app_id]
                    time_diff = (pd.to_datetime(target_patent['publn_date']) - pd.to_datetime(source_date)).days
                    
                    # Skip if time lag exceeds maximum
                    if time_diff > MAX_TIME_LAG_DAYS:
                        continue
                    
                    # Record similarity-based diffusion
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
                        'target_date': target_patent['publn_date'],
                        'novelty_category': novelty_category
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

def compute_international_diffusion(df, embeddings, novelty_category, batch_years=BATCH_YEARS):
    """
    Compute bidirectional international diffusion metrics (US<->EP patents).
    Uses parallel processing for different time batches and directions.
    
    Args:
        df: DataFrame of patents
        embeddings: Patent embeddings
        novelty_category: Which novelty category to analyze ('High', 'Medium' or 'Low')
        batch_years: Number of years to process in each batch
    """
    # Determine optimal number of workers with SLURM support
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    
    # Add year column for batching if it doesn't exist
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['publn_date']).dt.year
        
    year_min, year_max = df['year'].min(), df['year'].max()
    
    # Create arguments for each time batch
    batch_args = []
    for start_year in range(year_min, year_max, batch_years):
        end_year = start_year + batch_years
        # Add arguments for both directions
        batch_args.append((start_year, end_year, df, embeddings, novelty_category, 'US', 'EP'))
        batch_args.append((start_year, end_year, df, embeddings, novelty_category, 'EP', 'US'))
    
    print(f"Processing {len(batch_args)} time batches for {novelty_category} novelty patents using {num_workers} workers")
    
    # Process batches in parallel
    with mp.Pool(processes=min(len(batch_args), num_workers)) as pool:
        results_list = list(tqdm_auto(
            pool.imap(process_international_batch_directional, batch_args),
            total=len(batch_args),
            desc=f"Processing time batches for bidirectional international diffusion ({novelty_category} novelty)"
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
                                       f'international_diffusion_{source}_to_{target}_citation_{novelty_category.lower()}_novelty.csv')
            group.to_csv(citation_file, index=False)

    # Save semantic results
    if all_semantic_results:
        semantic_df = pd.DataFrame(all_semantic_results)
        for (source, target), group in semantic_df.groupby(['source_auth', 'target_auth']):
            semantic_file = os.path.join(output_dir, 'semantic',
                                       f'international_diffusion_{source}_to_{target}_semantic_{novelty_category.lower()}_novelty.csv')
            group.to_csv(semantic_file, index=False)
    
    # Return combined results for visualization
    return pd.DataFrame(all_citation_results + all_semantic_results)

def create_comparative_visualizations(diffusion_results_by_category):
    """Create visualizations comparing high vs. low novelty patent diffusion."""
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Diffusion time comparison
    fig = go.Figure()
    
    for novelty_category in diffusion_results_by_category.keys():
        df = diffusion_results_by_category[novelty_category]
        for diff_type in df['diffusion_type'].unique():
            subset = df[df['diffusion_type'] == diff_type]
            fig.add_trace(go.Histogram(
                x=subset['time_lag_days'] / 365.25,
                name=f"{novelty_category} Novelty - {diff_type}",
                opacity=0.7,
                nbinsx=20
            ))
    
    fig.update_layout(
        title='Distribution of Diffusion Time by Novelty Category and Diffusion Type',
        xaxis_title='Diffusion Time (Years)',
        yaxis_title='Count',
        barmode='overlay',
        legend_title='Category & Type'
    )
    
    fig.write_html('visualizations/diffusion_time_by_novelty_comparison.html')
    
    # 2. Average diffusion time by Y02 class and novelty
    all_data = []
    for novelty_category, df in diffusion_results_by_category.items():
        # Calculate average diffusion time by Y02 class
        for diff_type in df['diffusion_type'].unique():
            type_df = df[df['diffusion_type'] == diff_type]
            avg_times = type_df.groupby('source_class')['time_lag_days'].mean().reset_index()
            avg_times['time_lag_years'] = avg_times['time_lag_days'] / 365.25
            avg_times['novelty_category'] = novelty_category
            avg_times['diffusion_type'] = diff_type
            all_data.append(avg_times)
    
    if all_data:
        combined_avg_times = pd.concat(all_data)
        
        fig = px.bar(
            combined_avg_times,
            x='source_class',
            y='time_lag_years',
            color='novelty_category',
            facet_row='diffusion_type',
            barmode='group',
            title='Average Diffusion Speed by Y02 Class, Diffusion Type, and Novelty Category',
            labels={'time_lag_years': 'Average Diffusion Time (Years)', 'source_class': 'Y02 Class'},
            text_auto='.1f'
        )
        
        fig.update_layout(
            legend_title='Novelty Category',
            xaxis_tickangle=-45
        )
        
        fig.write_html('visualizations/diffusion_speed_by_novelty_comparison.html')
    
    # 3. Citation vs. Similarity comparison
    all_data = []
    for novelty_category, df in diffusion_results_by_category.items():
        df = df.copy()
        df['diffusion_mechanism'] = df['has_citation'].map({True: 'Citation', False: 'Semantic Similarity'})
        mechanism_counts = df.groupby(['diffusion_type', 'diffusion_mechanism']).size().reset_index(name='count')
        mechanism_counts['novelty_category'] = novelty_category
        all_data.append(mechanism_counts)
    
    if all_data:
        combined_mechanisms = pd.concat(all_data)
        
        fig = px.bar(
            combined_mechanisms, 
            x='diffusion_type', 
            y='count', 
            color='diffusion_mechanism',
            facet_col='novelty_category',
            title='Diffusion Mechanisms by Type and Novelty Category',
            labels={'diffusion_type': 'Diffusion Type', 'count': 'Number of Connections'},
            barmode='group'
        )
        
        fig.write_html('visualizations/diffusion_mechanisms_by_novelty.html')
    
    # 4. Comparative time series
    all_data = []
    for novelty_category, df in diffusion_results_by_category.items():
        df = df.copy()
        df['year'] = pd.to_datetime(df['target_date']).dt.year
        time_series = df.groupby(['year', 'diffusion_type']).size().reset_index(name='count')
        time_series['novelty_category'] = novelty_category
        all_data.append(time_series)
    
    if all_data:
        combined_time_series = pd.concat(all_data)
        
        fig = go.Figure()
        
        for novelty_category in combined_time_series['novelty_category'].unique():
            for diff_type in combined_time_series['diffusion_type'].unique():
                data = combined_time_series[
                    (combined_time_series['novelty_category'] == novelty_category) & 
                    (combined_time_series['diffusion_type'] == diff_type)
                ]
                
                fig.add_trace(go.Scatter(
                    x=data['year'],
                    y=data['count'],
                    mode='lines+markers',
                    name=f"{novelty_category} Novelty - {diff_type}",
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='TEK Diffusion Over Time by Type and Novelty Category',
            xaxis_title='Year',
            yaxis_title='Number of Diffusion Events',
            hovermode='x unified',
            legend_title='Category & Type'
        )
        
        fig.write_html('visualizations/diffusion_time_series_by_novelty.html')
    
    print("Comparative visualizations saved to 'visualizations' directory.")

def analyze_diffusion_by_novelty():
    """Main function to analyze diffusion patterns by novelty category using pre-computed embeddings."""
    # Get start time for performance tracking
    start_time = datetime.now()
    
    # Check available CPU resources
    num_cores = mp.cpu_count()
    print(f"Running on a machine with {num_cores} CPU cores")
    
    # Check if required files exist
    if not os.path.exists(f'edv_tek_cleantech_patstat_diffusion_preprocessed_novelty_{MIN_YEAR}_{MAX_YEAR}.parquet'):
        print(f"Error: preprocessed data file not found.")
        print("Please run generate_embeddings.py first.")
        return
    
    if not os.path.exists(f'edv_tek_cleantech_patstat_diffusion_embeddings_novelty_{MIN_YEAR}_{MAX_YEAR}.npy'):
        print("Error: embeddings file not found.")
        print("Please run generate_embeddings.py first.")
        return
    
    # Load preprocessed data and embeddings
    print("Loading preprocessed data and embeddings...")
    base_df = pd.read_parquet(f'edv_tek_cleantech_patstat_diffusion_preprocessed_novelty_{MIN_YEAR}_{MAX_YEAR}.parquet')
    embeddings = np.load(f'edv_tek_cleantech_patstat_diffusion_embeddings_novelty_{MIN_YEAR}_{MAX_YEAR}.npy')
    
    # Verify alignment
    print(f"Base dataframe size: {len(base_df)}, Embeddings size: {len(embeddings)}")
    assert len(base_df) == len(embeddings), "Dataframe and embeddings must have the same length"
    base_df['embedding_idx'] = np.arange(len(base_df))
    
    # Check if we have novelty data in the base dataframe
    if 'novelty_category' not in base_df.columns:
        print("Novelty categories not found in the preprocessed data.")
        print("Please run generate_embeddings.py with novelty data.")
        return
    
    # Print novelty distribution
    print("Novelty category distribution:")
    print(base_df['novelty_category'].value_counts())
    
    # Store results for each novelty category
    diffusion_results_by_category = {}
    
    # Process each novelty category separately
    for novelty_category in NOVELTY_CATEGORIES:
        print(f"\n{'='*80}")
        print(f"Processing {novelty_category} Novelty Patents")
        print(f"{'='*80}")
        
        # Filter base_df for the current novelty category
        novelty_df = base_df[base_df['novelty_category'] == novelty_category].copy()
        
        if len(novelty_df) == 0:
            print(f"No patents found with novelty category: {novelty_category}")
            continue
            
        print(f"Found {len(novelty_df)} patents with {novelty_category} novelty")
        
        # Explode the dataframe by Y02 classes
        print("Exploding dataframe by Y02 classes...")
        # Check the type of y02_classes and handle appropriately
        first_item = novelty_df['y02_classes'].iloc[0]
        if isinstance(first_item, np.ndarray):
            print(f"Note: y02_classes contains NumPy arrays, which is compatible with explode()")
        elif not isinstance(first_item, (list, tuple)):
            print(f"Warning: y02_classes contains type {type(first_item)}, which may cause issues")
            # Try to convert to list if possible
            try:
                print("Attempting to convert y02_classes to lists...")
                novelty_df['y02_classes'] = novelty_df['y02_classes'].apply(lambda x: list(x) if hasattr(x, '__iter__') else [x])
                print("Conversion successful")
            except Exception as e:
                print(f"Error converting to lists: {e}")

        # Handle empty classes
        novelty_df['y02_classes'] = novelty_df['y02_classes'].apply(lambda x: x if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 else [])

        # Now explode the dataframe
        df = novelty_df.explode('y02_classes').rename(columns={'y02_classes': 'y02_class'})
        
        # Remove rows with empty or null y02_class
        df = df[~pd.isna(df['y02_class'])].reset_index(drop=True)
        df = df[df['y02_class'] != ''].reset_index(drop=True)

        # Save original index for embedding reference
        df['original_index'] = df['embedding_idx']

        df = df.reset_index(drop=True)

        print(f"Exploded dataframe has {len(df)} patent-class combinations for {novelty_category} novelty")
        
        # Set up overall progress tracking
        with tqdm_auto(total=3, desc=f"Overall analysis progress ({novelty_category} novelty)", position=0) as pbar:
            # 1. Compute within-class diffusion
            print(f"\nComputing within-class diffusion for {novelty_category} novelty patents...")
            within_diffusion = compute_within_class_diffusion(df, embeddings, novelty_category)
            within_diffusion.to_csv(f'within_class_diffusion_{novelty_category.lower()}_novelty.csv', index=False)
            print(f"Found {len(within_diffusion)} within-class diffusion connections for {novelty_category} novelty")
            pbar.update(1)
            
            # 2. Compute between-class diffusion
            print(f"\nComputing between-class diffusion for {novelty_category} novelty patents...")
            between_diffusion = compute_between_class_diffusion(df, embeddings, novelty_category)
            between_diffusion.to_csv(f'between_class_diffusion_{novelty_category.lower()}_novelty.csv', index=False)
            print(f"Found {len(between_diffusion)} between-class diffusion connections for {novelty_category} novelty")
            pbar.update(1)
            
            # 3. Compute international diffusion
            print(f"\nComputing international diffusion for {novelty_category} novelty patents...")
            international_diffusion = compute_international_diffusion(df, embeddings, novelty_category)
            print(f"Found {len(international_diffusion)} international diffusion connections for {novelty_category} novelty")
            pbar.update(1)
        
        # Combine all diffusion types for this novelty category
        all_diffusion = pd.concat([within_diffusion, between_diffusion, international_diffusion])
        diffusion_results_by_category[novelty_category] = all_diffusion
        
        # 4. Compute summary statistics for this novelty category
        # Average diffusion time by type
        diffusion_time_stats = all_diffusion.groupby('diffusion_type')['time_lag_days'].agg(['mean', 'median', 'count'])
        print(f"\nAverage diffusion time by type for {novelty_category} novelty:")
        print(diffusion_time_stats)
        
        # Average diffusion time by Y02 class
        class_diffusion_stats = all_diffusion.groupby(['source_class', 'target_class'])['time_lag_days'].agg(['mean', 'median', 'count'])
        class_diffusion_stats = class_diffusion_stats.sort_values('count', ascending=False)
        
        print(f"\nTop class pairs by diffusion count for {novelty_category} novelty:")
        print(class_diffusion_stats.head(10))
        
        # Save combined results for this novelty category
        all_diffusion.to_csv(f'diffusion_all_{novelty_category.lower()}_novelty.csv', index=False)
    
    # Create comparative visualizations
    if len(diffusion_results_by_category) > 1:
        print("\nCreating comparative visualizations...")
        create_comparative_visualizations(diffusion_results_by_category)
    
    # Calculate and display total execution time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nAnalysis complete! Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze diffusion patterns for patents by novelty')
    parser.add_argument('--high-only', action='store_true', help='Only analyze high novelty patents')
    parser.add_argument('--low-only', action='store_true', help='Only analyze low novelty patents')
    args = parser.parse_args()
    
    if args.high_only:
        NOVELTY_CATEGORIES = ['High']
    elif args.low_only:
        NOVELTY_CATEGORIES = ['Low']
    
    analyze_diffusion_by_novelty()