import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import multiprocessing
from functools import partial
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_cosine_distance(vec1, vec2):
    """
    Compute the cosine distance between two vectors.
    Distance of 0 means identical, 2 means completely opposite.
    """
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_novelty_score(cited_papers_embeddings, q=100):
    """
    Compute the novelty score based on Shibayama et al. methodology.
    
    Args:
        cited_papers_embeddings: List of embeddings for papers cited by a patent
        q: Percentile value for aggregating distance scores (default=100, i.e., maximum)
        
    Returns:
        Novelty score or None if less than 2 papers are cited
    """
    n_papers = len(cited_papers_embeddings)
    
    if n_papers < 2:
        return None
    
    # Compute distances between all pairs of cited papers
    distances = []
    for i in range(n_papers):
        for j in range(i+1, n_papers):
            dist = compute_cosine_distance(cited_papers_embeddings[i], cited_papers_embeddings[j])
            distances.append(dist)
    
    # Compute the qth percentile value of distances
    if q == 100:
        return max(distances)
    else:
        return np.percentile(distances, q)


def process_patent_batch(patents_batch, paper_id_to_embedding, min_cited_papers, q_values):
    """
    Process a batch of patents and compute their novelty scores
    
    Args:
        patents_batch: List of (patent_id, cited_papers) tuples
        paper_id_to_embedding: Dictionary mapping from paper index to embedding
        min_cited_papers: Minimum number of papers a patent must cite to be included
        q_values: List of percentile values for novelty computation
        
    Returns:
        List of dictionaries with patent_id and novelty scores
    """
    results = []

    # Process each patent in the batch
    for patent_id, cited_papers in patents_batch:
        if len(cited_papers) < min_cited_papers:
            continue
            
        # Get embeddings for cited papers
        try:
            cited_papers_embeddings = [paper_id_to_embedding[paper_id] for paper_id in cited_papers]
            
            # Compute novelty score for each q value
            row = {'patent_id': patent_id}
            for q in q_values:
                score = compute_novelty_score(cited_papers_embeddings, q)
                row[f'novelty_q{q}'] = score
            
            results.append(row)
        except KeyError as e:
            # Skip patents with missing embeddings and log the issue
            print(f"Warning: Skipping patent {patent_id} due to missing embedding for paper {e}")
            continue
        except Exception as e:
            print(f"Error processing patent {patent_id}: {str(e)}")
            continue
        
    return results


def compute_patent_novelty(h5_file_path, min_cited_papers=3, q_values=None, n_cores=32, batch_size=50, 
                    checkpoint_file=None, checkpoint_interval=10):
    """
    Compute novelty scores for patents based on semantic distances between cited papers.
    
    Args:
        h5_file_path: Path to the HDF5 file containing patents and papers data
        min_cited_papers: Minimum number of papers a patent must cite to be included
        q_values: List of percentile values for novelty computation (default: [100, 99, 95, 90, 80, 50])
        n_cores: Number of CPU cores to use for multiprocessing
        batch_size: Number of patents to process in each batch (affects progress bar granularity)
        checkpoint_file: File path to save checkpoints during processing
        checkpoint_interval: Save a checkpoint after processing this many batches
        
    Returns:
        DataFrame with patent IDs and their novelty scores
    """
    if q_values is None:
        q_values = [100, 99, 95, 90, 80, 50]
    
    print(f"Loading data from {h5_file_path}...")
    with h5py.File(h5_file_path, 'r') as f:
        # Load paper embeddings
        print("Loading paper embeddings...")
        paper_embeddings = f['paper_embeddings'][:]
        print(f"Loaded {len(paper_embeddings)} paper embeddings")
        
        # Load patent-paper citations
        print("Loading patent-paper citations...")
        patent_paper_citations = f['patent_paper_citations'][:]
        print(f"Loaded {len(patent_paper_citations)} patent-paper citation pairs")
    
    # Create dictionary mapping from paper index to embedding
    print("Processing paper embeddings...")
    paper_id_to_embedding = {}
    for i, embedding in enumerate(tqdm(paper_embeddings, desc="Creating embedding lookup")):
        paper_id_to_embedding[i] = embedding
    
    # Create dictionary grouping cited papers by patent
    print("Grouping cited papers by patent...")
    patent_to_cited_papers = {}
    for patent_id, paper_id in tqdm(patent_paper_citations, desc="Organizing citations"):
        if patent_id not in patent_to_cited_papers:
            patent_to_cited_papers[patent_id] = []
        patent_to_cited_papers[patent_id].append(paper_id)
    
    # Filter patents with fewer than min_cited_papers cited papers
    print(f"Filtering patents with at least {min_cited_papers} cited papers...")
    patents_with_enough_citations = {}
    for patent_id, cited_papers in tqdm(patent_to_cited_papers.items(), desc="Filtering patents"):
        if len(cited_papers) >= min_cited_papers:
            patents_with_enough_citations[patent_id] = cited_papers
    
    # Convert dictionary to list of tuples for easier batching
    patents_list = list(patents_with_enough_citations.items())
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_patent_batch, 
        paper_id_to_embedding=paper_id_to_embedding,
        min_cited_papers=min_cited_papers,
        q_values=q_values
    )
    
    # Create smaller batches for more granular progress tracking
    patent_batches = []
    for i in range(0, len(patents_list), batch_size):
        batch = patents_list[i:i+batch_size]
        if batch:  # Ensure no empty batches
            patent_batches.append(batch)
    
    print(f"Computing novelty scores for {len(patents_with_enough_citations)} patents using {n_cores} cores...")
    print(f"Processing {len(patent_batches)} batches with ~{batch_size} patents per batch")
    
    # Check if we should resume from a checkpoint
    all_results = []
    start_batch = 0
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            checkpoint_df = pd.read_csv(checkpoint_file)
            all_results = checkpoint_df.to_dict('records')
            processed_patents = set(checkpoint_df['patent_id'].values)
            
            # Filter out already processed patents
            remaining_batches = []
            for batch in patent_batches:
                remaining_batch = []
                for patent_id, cited_papers in batch:
                    if patent_id not in processed_patents:
                        remaining_batch.append((patent_id, cited_papers))
                if remaining_batch:
                    remaining_batches.append(remaining_batch)
            
            patent_batches = remaining_batches
            print(f"Resuming from checkpoint: {len(all_results)} patents already processed, {len(patent_batches)} batches remaining")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Starting from scratch")
    
    # Process the batches in parallel
    with multiprocessing.Pool(processes=n_cores) as pool:
        # Use imap_unordered for potentially better performance
        batch_results_iter = pool.imap_unordered(process_func, patent_batches)
        
        # Set up progress bar
        pbar = tqdm(
            total=len(patent_batches), 
            desc="Processing patent batches",
            unit="batch"
        )
        
        batch_count = 0
        for batch_result in batch_results_iter:
            all_results.extend(batch_result)
            batch_count += 1
            pbar.update(1)
            
            # Save checkpoint if needed
            if checkpoint_file and batch_count % checkpoint_interval == 0:
                temp_df = pd.DataFrame(all_results)
                temp_df.to_csv(f"{checkpoint_file}_temp", index=False)
                # Atomic rename to prevent corruption
                os.rename(f"{checkpoint_file}_temp", checkpoint_file)
                pbar.write(f"Checkpoint saved: {len(all_results)} patents processed")
        
        pbar.close()
    
    # Create DataFrame with novelty scores
    novelty_df = pd.DataFrame(all_results)
    
    print(f"Novelty scores computed for {len(novelty_df)} patents.")
    
    # Save final checkpoint
    if checkpoint_file:
        novelty_df.to_csv(checkpoint_file, index=False)
        print(f"Final checkpoint saved to {checkpoint_file}")
        
    return novelty_df


def plot_novelty_distribution(novelty_df, q_values=None, save_html=True):
    """
    Plot the distribution of novelty scores for different q values.
    """
    
    if q_values is None:
        q_values = [100, 99, 95, 90, 80, 50]
    
    # Create subplots
    fig = make_subplots(rows=len(q_values), cols=1, 
                        subplot_titles=[f'Distribution of Novelty Scores (q={q})' for q in q_values])
    
    for i, q in enumerate(q_values):
        col = f'novelty_q{q}'
        
        # Create histogram
        fig.add_trace(
            go.Histogram(x=novelty_df[col], opacity=0.7, nbinsx=50),
            row=i+1, col=1
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Novelty Score", row=i+1, col=1)
        fig.update_yaxes(title_text="Frequency", row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        height=300*len(q_values),
        width=1000,
        showlegend=False
    )
    
    # Save as PNG
    fig.write_image("novelty_distributions.png")
    
    # Save as HTML if requested
    if save_html:
        fig.write_html("novelty_distributions.html")


def main():
    parser = argparse.ArgumentParser(description='Compute patent novelty based on semantic distances between cited papers')
    parser.add_argument('--h5_file', type=str, default='edv_tek_emergence_gnn_dataset.h5', 
                        help='Path to HDF5 file containing patents and papers data')
    parser.add_argument('--min_cited_papers', type=int, default=2,
                        help='Minimum number of papers a patent must cite to be included')
    parser.add_argument('--output', type=str, default='edv_tek_diffusion_patent_novelty_scores.csv',
                        help='Output file for novelty scores')
    parser.add_argument('--cores', type=int, default=32,
                        help='Number of CPU cores to use for multiprocessing')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of patents to process in each batch (smaller = more progress updates)')
    parser.add_argument('--q_values', type=str, default='100,99,95,90,80,50',
                        help='Comma-separated list of percentile values for novelty computation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='File path to save checkpoints during processing')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Save a checkpoint after processing this many batches')
    
    args = parser.parse_args()
    
    # Parse q_values from string to list
    q_values = [int(q) for q in args.q_values.split(',')]
    
    # Set the number of cores to use
    n_cores = min(args.cores, multiprocessing.cpu_count())
    print(f"Using {n_cores} CPU cores out of {multiprocessing.cpu_count()} available")
    print(f"Using batch size of {args.batch_size} patents for progress tracking")
    print(f"Computing novelty with q-values: {q_values}")
    
    # Record start time for performance tracking
    import time
    start_time = time.time()
    
    try:
        # Compute novelty scores
        novelty_df = compute_patent_novelty(
            args.h5_file, 
            args.min_cited_papers, 
            q_values=q_values, 
            n_cores=n_cores,
            batch_size=args.batch_size,
            checkpoint_file=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Save to CSV
        novelty_df.to_csv(args.patent_novelty_scores.csv, index=False)
        print(f"Novelty scores saved to {args.output}")
        
        # Plot distributions
        plot_novelty_distribution(novelty_df, q_values=q_values)
        print(f"Novelty distributions saved as PNG and HTML")
        
        # Report total execution time
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()