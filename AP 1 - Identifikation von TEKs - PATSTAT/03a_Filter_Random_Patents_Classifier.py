import pandas as pd
import numpy as np
from sqlalchemy import create_engine, URL, text
import psycopg2
import psycopg2.extras
import os
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Start timing
start_time = time.time()

# Database connection parameters
url_object = URL.create(
    drivername="",
    username="",
    password="",
    host="",
    port="",
    database=""
)
engine = create_engine(url_object, pool_size=10, max_overflow=20)

# Direct psycopg2 connection for more efficient batch operations
conn_params = {
    "user": "",
    "password": "",
    "host": "",
    "port": "",
    "database": ""
}

def get_direct_connection():
    """Create a direct psycopg2 connection with optimized parameters"""
    conn = psycopg2.connect(
        **conn_params,
        cursor_factory=psycopg2.extras.DictCursor
    )
    # Optimize connection settings
    with conn.cursor() as cursor:
        cursor.execute("SET work_mem = '256MB'")
    return conn

# Output path
output_dir = Path('/mnt/hdd02/Projekt_EDV_TEK')
output_file = output_dir / 'edv_tek_non_cleantech_citation_related_title_abstract.csv'

# Target limits - set explicit limits to manage data volume
TOTAL_PATENTS_LIMIT = 3000000  # Target total patents
CITED_BY_CLEANTECH_LIMIT = 1500000  # Limit for patents cited by cleantech
CITING_CLEANTECH_LIMIT = 1500000  # Limit for patents citing cleantech

# 1. Read cleantech application IDs
logger.info("Reading cleantech patents application IDs...")
cleantech_df = pd.read_csv(
    output_dir / 'edv_tek_all_cleantech_appln_ids.csv',
    usecols=['appln_id']
)
cleantech_patent_count = len(cleantech_df)
logger.info(f"Found {cleantech_patent_count} cleantech patents")

# 2. Process citation relationships in stages to manage memory
with get_direct_connection() as conn:
    with conn.cursor() as cursor:
        # Start transaction
        cursor.execute("BEGIN;")
        
        # Create temporary table for cleantech appln_ids
        logger.info("Creating temporary table for cleantech patents...")
        cursor.execute("DROP TABLE IF EXISTS temp_cleantech_appln_ids;")
        cursor.execute("""
            CREATE TABLE temp_cleantech_appln_ids (
                appln_id TEXT PRIMARY KEY
            );
        """)
        
        # Insert cleantech appln_ids in batches
        psycopg2.extras.execute_values(
            cursor,
            "INSERT INTO temp_cleantech_appln_ids (appln_id) VALUES %s",
            [(str(appln_id),) for appln_id in cleantech_df['appln_id']],
            page_size=10000
        )
        
        # Create index on cleantech appln_ids
        cursor.execute("CREATE INDEX ON temp_cleantech_appln_ids (appln_id);")
        cursor.execute("ANALYZE temp_cleantech_appln_ids;")
        
        # Create table for cleantech publication IDs (much smaller and more efficient than a full join)
        logger.info("Creating table for cleantech publication IDs...")
        cursor.execute("""
            CREATE TABLE cleantech_pat_publn AS
            SELECT pub.pat_publn_id
            FROM tls211_pat_publn pub
            JOIN temp_cleantech_appln_ids ct ON pub.appln_id::text = ct.appln_id;
            
            CREATE INDEX ON cleantech_pat_publn (pat_publn_id);
            ANALYZE cleantech_pat_publn;
        """)
        
        # Create results table for citation relationships
        # Using a regular table instead of temporary to ensure persistence across connections
        cursor.execute("DROP TABLE IF EXISTS non_cleantech_citation_data;")
        cursor.execute("""
            CREATE TABLE non_cleantech_citation_data (
                appln_id TEXT PRIMARY KEY,
                cited_by_cleantech BOOLEAN DEFAULT FALSE,
                cites_cleantech BOOLEAN DEFAULT FALSE
            );
        """)
        
        # Step 1: Find patents CITED BY cleantech patents (with limit)
        logger.info(f"Finding patents cited by cleantech patents (limit: {CITED_BY_CLEANTECH_LIMIT})...")
        cursor.execute(f"""
            INSERT INTO non_cleantech_citation_data (appln_id, cited_by_cleantech)
            SELECT DISTINCT t211_cited.appln_id, TRUE
            FROM tls212_citation cit
            JOIN cleantech_pat_publn cp ON cit.pat_publn_id = cp.pat_publn_id
            JOIN tls211_pat_publn t211_cited ON cit.cited_pat_publn_id = t211_cited.pat_publn_id
            JOIN tls201_appln t201 ON t211_cited.appln_id = t201.appln_id
            WHERE t201.appln_auth IN ('US', 'EP')
              AND t201.granted = 'Y'
              AND t211_cited.appln_id::text NOT IN (SELECT appln_id FROM temp_cleantech_appln_ids)
            LIMIT {CITED_BY_CLEANTECH_LIMIT};
        """)
        
        cursor.execute("SELECT COUNT(*) FROM non_cleantech_citation_data WHERE cited_by_cleantech = TRUE")
        cited_count = cursor.fetchone()[0]
        logger.info(f"Found {cited_count} non-cleantech patents cited by cleantech patents")
        
        # Step 2: Find patents CITING cleantech patents (with limit)
        logger.info(f"Finding patents citing cleantech patents (limit: {CITING_CLEANTECH_LIMIT})...")
        cursor.execute(f"""
            INSERT INTO non_cleantech_citation_data (appln_id, cites_cleantech)
            SELECT DISTINCT t211_citing.appln_id, TRUE
            FROM tls212_citation cit
            JOIN cleantech_pat_publn cp ON cit.cited_pat_publn_id = cp.pat_publn_id
            JOIN tls211_pat_publn t211_citing ON cit.pat_publn_id = t211_citing.pat_publn_id
            JOIN tls201_appln t201 ON t211_citing.appln_id = t201.appln_id
            WHERE t201.appln_auth IN ('US', 'EP')
              AND t201.granted = 'Y'
              AND t211_citing.appln_id::text NOT IN (SELECT appln_id FROM temp_cleantech_appln_ids)
              AND t211_citing.appln_id::text NOT IN (SELECT appln_id FROM non_cleantech_citation_data)
            LIMIT {CITING_CLEANTECH_LIMIT};
        """)
        
        # For patents already in the table, update the cites_cleantech flag
        cursor.execute(f"""
            UPDATE non_cleantech_citation_data
            SET cites_cleantech = TRUE
            WHERE appln_id IN (
                SELECT DISTINCT t211_citing.appln_id
                FROM tls212_citation cit
                JOIN cleantech_pat_publn cp ON cit.cited_pat_publn_id = cp.pat_publn_id
                JOIN tls211_pat_publn t211_citing ON cit.pat_publn_id = t211_citing.pat_publn_id
                JOIN tls201_appln t201 ON t211_citing.appln_id = t201.appln_id
                WHERE t201.appln_auth IN ('US', 'EP')
                  AND t201.granted = 'Y'
                  AND t211_citing.appln_id::text NOT IN (SELECT appln_id FROM temp_cleantech_appln_ids)
                  AND t211_citing.appln_id::text IN (SELECT appln_id FROM non_cleantech_citation_data)
                LIMIT {CITING_CLEANTECH_LIMIT}
            );
        """)
        
        # Get count statistics
        cursor.execute("""
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN cited_by_cleantech AND cites_cleantech THEN 1 ELSE 0 END) AS both_count,
                SUM(CASE WHEN cited_by_cleantech AND NOT cites_cleantech THEN 1 ELSE 0 END) AS only_cited_count,
                SUM(CASE WHEN NOT cited_by_cleantech AND cites_cleantech THEN 1 ELSE 0 END) AS only_citing_count
            FROM non_cleantech_citation_data
        """)
        stats = cursor.fetchone()
        total_count, both_count, only_cited_count, only_citing_count = stats
        
        logger.info(f"Citation relationships found:")
        logger.info(f"- Total patents: {total_count}")
        logger.info(f"- {both_count} patents in both groups (bidirectional)")
        logger.info(f"- {only_cited_count} patents only cited by cleantech")
        logger.info(f"- {only_citing_count} patents only citing cleantech")
        
        # If we have more than our target limit, create a balanced sample
        if total_count > TOTAL_PATENTS_LIMIT:
            # Keep all bidirectional citations (both_count)
            # Calculate how many to keep from each one-way citation group
            remaining_quota = TOTAL_PATENTS_LIMIT - both_count
            per_group = remaining_quota // 2
            
            logger.info(f"Limiting dataset to {TOTAL_PATENTS_LIMIT} patents with balanced sampling...")
            
            # Create balanced dataset with random sampling
            cursor.execute(f"""
                CREATE TABLE balanced_citation_data AS
                -- Keep all patents that are in both groups
                SELECT * FROM non_cleantech_citation_data
                WHERE cited_by_cleantech AND cites_cleantech
                
                UNION ALL
                
                -- Sample from patents only cited by cleantech
                SELECT * FROM (
                    SELECT * FROM non_cleantech_citation_data
                    WHERE cited_by_cleantech AND NOT cites_cleantech
                    ORDER BY random()
                    LIMIT {per_group}
                ) cited_only
                
                UNION ALL
                
                -- Sample from patents only citing cleantech
                SELECT * FROM (
                    SELECT * FROM non_cleantech_citation_data
                    WHERE NOT cited_by_cleantech AND cites_cleantech
                    ORDER BY random()
                    LIMIT {per_group}
                ) citing_only;
            """)
            
            # Replace original table with balanced table
            cursor.execute("""
                DROP TABLE non_cleantech_citation_data;
                ALTER TABLE balanced_citation_data RENAME TO non_cleantech_citation_data;
            """)
            
            # Add index for performance
            cursor.execute("""
                CREATE INDEX ON non_cleantech_citation_data (appln_id);
                ANALYZE non_cleantech_citation_data;
            """)
            
            # Get updated counts
            cursor.execute("""
                SELECT COUNT(*) FROM non_cleantech_citation_data;
            """)
            balanced_count = cursor.fetchone()[0]
            logger.info(f"Created balanced dataset with {balanced_count} patents")
        
        # Commit all database changes
        cursor.execute("COMMIT;")

# 3. Extract patent metadata in efficient batches
logger.info("Extracting metadata for the selected patents...")

# Metadata extraction is now handled directly with server-side cursor

# Use more efficient server-side cursor approach for metadata extraction
results = []
batch_num = 0
total_processed = 0

logger.info("Using server-side cursor for efficient metadata extraction...")
with get_direct_connection() as conn:
    # Use named server-side cursor for efficient streaming of results
    with conn.cursor("server_side_cursor") as cursor:
        cursor.execute("""
            SELECT 
                t201.appln_id,
                t201.appln_auth,
                t202.appln_title,
                t203.appln_abstract,
                ncc.cited_by_cleantech,
                ncc.cites_cleantech
            FROM non_cleantech_citation_data ncc
            JOIN tls201_appln t201 ON ncc.appln_id = t201.appln_id::text
            JOIN tls202_appln_title t202 ON t201.appln_id = t202.appln_id AND t202.appln_title_lg = 'en'
            JOIN tls203_appln_abstr t203 ON t201.appln_id = t203.appln_id AND t203.appln_abstract_lg = 'en'
            WHERE t203.appln_abstract IS NOT NULL
        """)
        
        # Process in batches
        batch_size = 5000
        while True:
            batch_start = time.time()
            records = cursor.fetchmany(batch_size)
            
            if not records:
                break
                
            # Convert to DataFrame
            columns = [desc[0] for desc in cursor.description]
            batch_df = pd.DataFrame(records, columns=columns)
            total_processed += len(batch_df)
            
            # Save intermediary results to disk to manage memory
            batch_file = output_dir / f"batch_{batch_num}.csv"
            batch_df.to_csv(batch_file, index=False)
            
            batch_elapsed = time.time() - batch_start
            logger.info(f"Batch {batch_num}: Processed {len(batch_df)} patents in {batch_elapsed:.2f} seconds")
            
            # Track batch file path instead of keeping dataframe in memory
            results.append(batch_file)
            batch_num += 1
            
            # Report progress periodically
            if batch_num % 10 == 0:
                logger.info(f"Progress: {total_processed} patents processed so far")

# 4. Combine all batch files into final output
logger.info(f"Combining batches into final output: {output_file}")

# Use pandas to combine saved batch files
df_chunks = []
total_records = 0

for batch_file in results:
    chunk = pd.read_csv(batch_file)
    total_records += len(chunk)
    df_chunks.append(chunk)
    
    # Remove batch file after reading it
    os.remove(batch_file)

df_result = pd.concat(df_chunks, ignore_index=True)

# Final statistics
cited_count = df_result['cited_by_cleantech'].sum()
citing_count = df_result['cites_cleantech'].sum()
both_count = df_result[df_result['cited_by_cleantech'] & df_result['cites_cleantech']].shape[0]
only_cited_count = df_result[df_result['cited_by_cleantech'] & ~df_result['cites_cleantech']].shape[0]
only_citing_count = df_result[~df_result['cited_by_cleantech'] & df_result['cites_cleantech']].shape[0]

logger.info(f"Final dataset statistics:")
logger.info(f"- Total patents with metadata: {len(df_result)}")
logger.info(f"- {cited_count} cited by cleantech, {citing_count} citing cleantech")
logger.info(f"- {both_count} in both groups, {only_cited_count} only cited by cleantech, {only_citing_count} only citing cleantech")

# Save final output
logger.info(f"Saving final dataset to {output_file}")
df_result.to_csv(output_file, index=False)

# Clean up database temporary tables
with get_direct_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("""
            DROP TABLE IF EXISTS temp_cleantech_appln_ids;
            DROP TABLE IF EXISTS cleantech_pat_publn;
            DROP TABLE IF EXISTS non_cleantech_citation_data;
        """)

elapsed_time = time.time() - start_time
logger.info(f"Process completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")