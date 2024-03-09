import os
import pandas as pd
import logging
from tqdm import tqdm
from sqlalchemy import create_engine, URL, text

tqdm.pandas()

# Database connection details
url_object = URL.create(
    drivername='postgresql+psycopg2',
    username=user,
    password=password,
    host=host,
    port=port,
    database=database,
)
engine = create_engine(url_object) #, echo=True)

errors = []

# Function to process and upload each TSV file
def process_upload_tsv(file_path):
    global errors  # Declare errors as a global variable

    # Read TSV file into DataFrame
    df = pd.read_csv(file_path, compression='zip', sep='\t', header=0)

    # Cast patent_id column to string
    df['patent_id'] = df['patent_id'].astype(str)

    # For each row in df, match patent_id with publn_nr in tls211_pat_publn where publn_auth is "US"
    with engine.connect() as conn:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                patent_id = row['patent_id']
                query = text("""
                    SELECT appln_id FROM public.tls211_pat_publn
                    WHERE publn_nr = :patent_id AND publn_auth = 'US'
                """)
                result = conn.execute(query, {'patent_id': patent_id})
                appln_id = result.fetchone()

                appln_id = str(appln_id[0]).split('.')[0] if appln_id else None

                if appln_id:
                    df.at[index, 'appln_id'] = appln_id[0]
            except Exception as e:
                errors.append({'patent_id': patent_id, 'error': str(e)})

    # Create a new table or append to an existing table in the database
    df['publn_auth'] = 'US'
    df.to_sql('us_description', con=engine, if_exists='append', index=False)

    # Print the number of errors after processing this file
    print(f"Finished processing {file_path}. Number of errors: {len(errors)}. Number of successes: {len(df) - len(errors)}")

# Directory containing your TSV files
directory = '/mnt/hdd01/patentsview/Fulltext Data/Description'

# Process each file
for filename in os.listdir(directory):
    if filename.endswith('.tsv.zip'):
        process_upload_tsv(os.path.join(directory, filename))

print("Data upload complete.")

# Save errors to a CSV file
df_errors = pd.DataFrame(errors)
df_errors.to_csv('/mnt/hdd01/patentsview/Fulltext Data/Description/PATSTAT_2023_parsing_errors.csv', index=False)
