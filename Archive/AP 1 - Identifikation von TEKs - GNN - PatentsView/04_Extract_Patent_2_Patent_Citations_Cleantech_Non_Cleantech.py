from sqlalchemy import create_engine, URL, text, String
import time
import pandas as pd
import os

# Initialize database connection (update credentials as needed)
url_object = URL.create(
    drivername="",
    username="",
    password="",
    host="",
    port="",
    database=""
)
engine = create_engine(url_object)

# Output file path
output_file = '/mnt/hdd02/Projekt_EDV_TEK/edv_tek_cleantech_to_non_cleantech_citations.csv'

# Main process
def extract_citations():
    try:
        start_time = time.time()
        print("Starting citation extraction process...")
    
        # Load only application IDs to reduce memory usage and ensure TEXT format
        print("Loading application IDs...")
        df_cleantech = pd.read_csv('/mnt/hdd02/Projekt_EDV_TEK/edv_tek_all_cleantech_appln_ids.csv', usecols=['appln_id'])
        df_non_cleantech = pd.read_csv('/mnt/hdd02/Projekt_EDV_TEK/edv_tek_non_cleantech_all_title_abstract.csv', usecols=['appln_id'])
        
        # Convert appln_id columns to text/string type
        df_cleantech['appln_id'] = df_cleantech['appln_id'].astype(str)
        df_non_cleantech['appln_id'] = df_non_cleantech['appln_id'].astype(str)
        
        # Remove any rows with empty strings if any
        df_cleantech = df_cleantech[df_cleantech['appln_id'] != '']
        df_non_cleantech = df_non_cleantech[df_non_cleantech['appln_id'] != '']
        
        with engine.begin() as conn:
            # Create temporary tables with appropriate data types explicitly defined
            print("Creating temporary tables...")
            # Use dtype to explicitly set the datatype as TEXT
            df_cleantech.to_sql('temp_cleantech', conn, if_exists='replace', index=False, 
                            dtype={'appln_id': String})
            df_non_cleantech.to_sql('temp_non_cleantech', conn, if_exists='replace', index=False,
                                dtype={'appln_id': String})
            
            # Create indices for faster joins
            print("Creating indices...")
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_temp_cleantech ON temp_cleantech(appln_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_temp_non_cleantech ON temp_non_cleantech(appln_id)"))
            
            # Optimized query using CTEs to pre-filter publications
            print("Executing query...")
            query = """
            WITH cleantech_pubs AS (
                SELECT p.pat_publn_id, p.appln_id 
                FROM tls211_pat_publn p
                JOIN temp_cleantech c ON p.appln_id::text = c.appln_id
            ),
            non_cleantech_pubs AS (
                SELECT p.pat_publn_id, p.appln_id 
                FROM tls211_pat_publn p
                JOIN temp_non_cleantech nc ON p.appln_id::text = nc.appln_id
            )
            
            -- Cleantech citing Non-cleantech
            SELECT DISTINCT 
                c.pat_publn_id AS pat_publn_id,
                c.cited_pat_publn_id AS cited_pat_publn_id,
                cp.appln_id AS citing_appln_id,
                ncp.appln_id AS cited_appln_id
            FROM 
                tls212_citation c
            JOIN 
                cleantech_pubs cp ON c.pat_publn_id = cp.pat_publn_id
            JOIN 
                non_cleantech_pubs ncp ON c.cited_pat_publn_id = ncp.pat_publn_id
            
            UNION ALL
            
            -- Non-cleantech citing Cleantech
            SELECT DISTINCT 
                c.pat_publn_id AS pat_publn_id,
                c.cited_pat_publn_id AS cited_pat_publn_id,
                ncp.appln_id AS citing_appln_id,
                cp.appln_id AS cited_appln_id
            FROM 
                tls212_citation c
            JOIN 
                non_cleantech_pubs ncp ON c.pat_publn_id = ncp.pat_publn_id
            JOIN 
                cleantech_pubs cp ON c.cited_pat_publn_id = cp.pat_publn_id
            """
            
            # Process in chunks to manage memory
            chunk_size = 100000
            offset = 0
            total_records = 0
            
            # Create or clear the output file
            open(output_file, 'w').close()
            header_written = False
            
            while True:
                print(f"Processing chunk at offset {offset}...")
                chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
                chunk_df = pd.read_sql_query(text(chunk_query), conn)
                
                if chunk_df.empty:
                    break
                    
                # Write to CSV
                chunk_df.to_csv(output_file, mode='a', header=not header_written, index=False)
                header_written = True
                
                records_processed = len(chunk_df)
                total_records += records_processed
                print(f"  Processed {records_processed} citations (total: {total_records})")
                
                offset += chunk_size
                
                # If this chunk was smaller than chunk_size, we've reached the end
                if records_processed < chunk_size:
                    break
            
            # Clean up
            print("Cleaning up temporary tables and indices...")
            conn.execute(text("""
            DROP INDEX IF EXISTS idx_temp_cleantech;
            DROP INDEX IF EXISTS idx_temp_non_cleantech;
            DROP TABLE IF EXISTS temp_cleantech;
            DROP TABLE IF EXISTS temp_non_cleantech;
            """))
        
            elapsed_time = time.time() - start_time
            print(f"Citation extraction complete. Processed {total_records} citations in {elapsed_time:.2f} seconds.")
            print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error during execution: {e}")
        # Make sure to clean up even if there's an error
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                DROP INDEX IF EXISTS idx_temp_cleantech;
                DROP INDEX IF EXISTS idx_temp_non_cleantech;
                DROP TABLE IF EXISTS temp_cleantech;
                DROP TABLE IF EXISTS temp_non_cleantech;
                """))
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        raise

# Run the process
if __name__ == "__main__":
    extract_citations()