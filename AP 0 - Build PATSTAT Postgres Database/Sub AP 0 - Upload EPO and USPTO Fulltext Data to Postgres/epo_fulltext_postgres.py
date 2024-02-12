import os
import pandas as pd
import logging
from tqdm import tqdm
from sqlalchemy import create_engine, URL, text

tqdm.pandas()

# Database connection details
url_object = URL.create(
    drivername='postgresql+psycopg2',
    username='tie',
    password='TIE%2023!tuhh',
    host='127.0.0.1',
    port=65432,
    database='PATSTAT_2023',
)
engine = create_engine(url_object) #, echo=True)

errors = []

# Function to process and upload each TSV file
def process_upload_tsv(file_path):
    global errors 

    # Read TSV file into DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, names=['appln_auth', 'epo_publn_nr', 'appln_kind', 'appln_date', 'appln_lng', 'appln_comp', 'appln_text_type', 'appln_text'])

    # Cast publn_nr column to string
    df['epo_publn_nr'] = df['epo_publn_nr'].astype(str)

    # For each row in df, match publn_nr with publn_nr in tls211_pat_publn where publn_auth is "US"
    with engine.connect() as conn:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                epo_publn_nr = row['epo_publn_nr']
                query = text("""
                    SELECT appln_id FROM public.tls211_pat_publn
                    WHERE publn_nr = :epo_publn_nr AND publn_auth = 'EP'
                """)
                result = conn.execute(query, {'epo_publn_nr': epo_publn_nr})
                appln_id = result.fetchone()

                appln_id = str(appln_id[0]).split('.')[0] if appln_id else None

                if appln_id:
                    df.at[index, 'appln_id'] = appln_id[0]
            except Exception as e:
                errors.append({'epo_publn_nr': epo_publn_nr, 'error': str(e)})

    # Create a new table or append to an existing table in the database
    df['publn_auth'] = 'EP'
    df.to_sql('ep_fulltext_data', con=engine, if_exists='append', index=False)

    # Print the number of errors after processing this file
    print(f"Finished processing {file_path}. Number of errors: {len(errors)}. Number of successes: {len(df) - len(errors)}")

# Directory containing your TSV files
directory = '/mnt/hdd01/EP_Fulltext_Data'

# Process each file
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        process_upload_tsv(os.path.join(directory, filename))

print("Data upload complete.")

# Save errors to a CSV file
df_errors = pd.DataFrame(errors)
df_errors.to_csv('/mnt/hdd01/EP_Fulltext_Data/PATSTAT_2023_fulltext_parsing_errors.csv', index=False)
