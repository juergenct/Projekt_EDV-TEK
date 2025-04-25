import os
import pandas as pd
from sqlalchemy import create_engine, URL
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from tqdm import tqdm

# Database connection details
url_object = URL.create(
    drivername="postgresql+psycopg2",
    username="user",
    password="password",
    host="localhost",
    port="25432",
    database="database",
)
engine = create_engine(url_object)  #, echo=True)

directory = '/mnt/hdd01/patentsview/Fulltext Data/Brief Summary'

for file in tqdm(os.listdir(directory), desc="Processing files"):
    if file.endswith('.tsv'):
        try:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            df['publn_auth'] = 'US'
            with engine.connect() as conn:
                try:
                    df.to_sql('us_brf_summary', conn, if_exists='append', index=False)
                except (SQLAlchemyError, OperationalError) as e:
                    print(f"Error inserting data from {file_path}: {e}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")