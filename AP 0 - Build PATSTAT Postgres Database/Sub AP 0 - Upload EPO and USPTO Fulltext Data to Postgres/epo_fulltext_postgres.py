import os
import pandas as pd
from sqlalchemy import create_engine, URL
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from multiprocessing import Pool
from tqdm import tqdm

# Database connection details
url_object = URL.create(
    drivername="",
    username="",
    password="",
    host="",
    port="",
    database=""
)
engine = create_engine(url_object) #, echo=True)
directory = '/mnt/hdd01/EP_Fulltext_Data'

for file in tqdm(os.listdir(directory), desc="Processing files"):
    if file.endswith('.txt'):
        try:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, sep='\t', header=None, names=['appln_auth', 'epo_publn_nr', 'appln_kind', 'appln_date', 'appln_lng', 'appln_comp', 'appln_text_type', 'appln_text'])
            df['epo_publn_nr'] = df['epo_publn_nr'].astype(str)
            df['publn_auth'] = 'EP'
            with engine.connect() as conn:
                try:
                    df.to_sql('ep_fulltext_data', conn, if_exists='append', index=False)
                except (SQLAlchemyError, OperationalError) as e:
                    print(f"Error inserting data from {file_path}: {e}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
