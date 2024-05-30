import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, URL
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
tqdm.pandas()

model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

url_object = URL.create(
    drivername="postgresql+psycopg2",
    username="tie",
    password="TIE%2023!tuhh",
    host="134.28.58.100",
    port="65432",
    database="PATSTAT_2023"
)
engine = create_engine(url_object)


df_patstat_cleantech_metadata = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_cleantech_granted_abstract_metadata.json', orient='records')
df_patstat_non_cleantech_metadata = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_non_cleantech_granted_abstract_metadata.json', orient='records')

df_patstat_us_cleantech_metadata = df_patstat_cleantech_metadata[df_patstat_cleantech_metadata['appln_auth'].apply(lambda x: 'US' in x or 'EP' in x)]
df_patstat_us_non_cleantech_metadata = df_patstat_non_cleantech_metadata[df_patstat_non_cleantech_metadata['appln_auth'].apply(lambda x: 'US' in x or 'EP' in x)]

print(f"Number of Cleantech Patents: {len(df_patstat_us_cleantech_metadata)}, Number of Non-Cleantech Patents: {len(df_patstat_us_non_cleantech_metadata)}")

### 
df_patstat_us_cleantech_metadata['appln_id'] = df_patstat_us_cleantech_metadata['appln_id'].astype(str)
df_patstat_us_non_cleantech_metadata['appln_id'] = df_patstat_us_non_cleantech_metadata['appln_id'].astype(str)

df_patstat_us_cleantech_metadata['appln_id'].to_sql('temp_patstat_cleantech_metadata', engine, if_exists='replace', index=False)
df_patstat_us_non_cleantech_metadata['appln_id'].to_sql('temp_patstat_non_cleantech_metadata', engine, if_exists='replace', index=False)

df_patstat_publn_id_cleantech = pd.read_sql_query("""
    SELECT tp.appln_id, pp.pat_publn_id
    FROM temp_patstat_cleantech_metadata AS tp
    JOIN tls211_pat_publn AS pp ON tp.appln_id = pp.appln_id
""", con=engine)
df_patstat_publn_id_cleantech = df_patstat_publn_id_cleantech.drop_duplicates(subset=['appln_id', 'pat_publn_id'])
df_patstat_publn_id_cleantech.to_sql('temp_patstat_cleantech_metadata', engine, if_exists='replace', index=False)

df_patstat_publn_id_non_cleantech = pd.read_sql_query("""
    SELECT tp.appln_id, pp.pat_publn_id
    FROM temp_patstat_non_cleantech_metadata AS tp
    JOIN tls211_pat_publn AS pp ON tp.appln_id = pp.appln_id
""", con=engine)
df_patstat_publn_id_non_cleantech = df_patstat_publn_id_non_cleantech.drop_duplicates(subset=['appln_id', 'pat_publn_id'])
df_patstat_publn_id_non_cleantech.to_sql('temp_patstat_non_cleantech_metadata', engine, if_exists='replace', index=False)

df_patstat_cleantech_citations = pd.read_sql_query("""
    SELECT c.pat_publn_id, c.cited_pat_publn_id, p.appln_id AS cited_appln_id
    FROM temp_patstat_cleantech_metadata AS tc
    JOIN tls212_citation AS c ON tc.pat_publn_id = c.pat_publn_id
    JOIN tls211_pat_publn AS p ON c.cited_pat_publn_id = p.pat_publn_id
""", con=engine)
df_patstat_cleantech_citations = df_patstat_cleantech_citations.drop_duplicates(subset=['pat_publn_id', 'cited_pat_publn_id', 'cited_appln_id'])

df_patstat_non_cleantech_citations = pd.read_sql_query("""
    SELECT c.pat_publn_id, c.cited_pat_publn_id, p.appln_id AS cited_appln_id
    FROM temp_patstat_non_cleantech_metadata AS tc
    JOIN tls212_citation AS c ON tc.pat_publn_id = c.pat_publn_id
    JOIN tls211_pat_publn AS p ON c.cited_pat_publn_id = p.pat_publn_id
""", con=engine)
df_patstat_non_cleantech_citations = df_patstat_non_cleantech_citations.drop_duplicates(subset=['pat_publn_id', 'cited_pat_publn_id', 'cited_appln_id'])

print(f"Number of Cleantech Citations: {len(df_patstat_cleantech_citations)}, Number of Non-Cleantech Citations: {len(df_patstat_non_cleantech_citations)}")

###
df_patstat_cleantech_citations_works = pd.DataFrame(df_patstat_cleantech_citations['cited_appln_id'].unique(), columns=['cited_appln_id'])
df_patstat_cleantech_citations_works.to_sql('temp_patstat_cleantech_citations', engine, if_exists='replace', index=False)
df_patstat_non_cleantech_citations_works = pd.DataFrame(df_patstat_non_cleantech_citations['cited_appln_id'].unique(), columns=['cited_appln_id'])
df_patstat_non_cleantech_citations_works.to_sql('temp_patstat_non_cleantech_citations', engine, if_exists='replace', index=False)


df_patstat_cleantech_citations_en = pd.read_sql_query("""
    SELECT tc.cited_appln_id, t2.appln_title, t3.appln_abstract
    FROM temp_patstat_cleantech_citations AS tc
    JOIN tls202_appln_title AS t2 ON tc.cited_appln_id = t2.appln_id AND t2.appln_title_lg = 'en'
    JOIN tls203_appln_abstr AS t3 ON tc.cited_appln_id = t3.appln_id AND t3.appln_abstract_lg = 'en'
""", con=engine)

df_patstat_non_cleantech_citations_en = pd.read_sql_query("""
    SELECT tc.cited_appln_id, t2.appln_title, t3.appln_abstract
    FROM temp_patstat_non_cleantech_citations AS tc
    JOIN tls202_appln_title AS t2 ON tc.cited_appln_id = t2.appln_id AND t2.appln_title_lg = 'en'
    JOIN tls203_appln_abstr AS t3 ON tc.cited_appln_id = t3.appln_id AND t3.appln_abstract_lg = 'en'
""", con=engine)

print(f"Number of Cleantech Citations Works: {len(df_patstat_cleantech_citations_en)}, Number of Non-Cleantech Citations Works: {len(df_patstat_non_cleantech_citations_en)}")

df_patstat_cleantech_citations_en.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_cleantech_citations_en_grouped.csv')
df_patstat_non_cleantech_citations_en.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_non_cleantech_citations_en_grouped.csv')

df_patstat_us_cleantech_metadata.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_us_cleantech_metadata.csv')
df_patstat_us_non_cleantech_metadata.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_us_non_cleantech_metadata.csv')

###
df_patstat_cleantech_citations_en['embedding'] = model.encode((df_patstat_cleantech_citations_en['appln_title'] + ' [SEP] ' + df_patstat_cleantech_citations_en['appln_abstract']), show_progress_bar=True).tolist()
df_patstat_non_cleantech_citations_en['embedding'] = model.encode((df_patstat_non_cleantech_citations_en['appln_title'] + ' [SEP] ' + df_patstat_non_cleantech_citations_en['appln_abstract']), show_progress_bar=True).tolist()

df_patstat_us_cleantech_metadata['appln_title'] = df_patstat_us_cleantech_metadata['appln_title'].apply(' '.join)
df_patstat_us_cleantech_metadata['appln_abstract'] = df_patstat_us_cleantech_metadata['appln_abstract'].apply(' '.join)
df_patstat_us_non_cleantech_metadata['appln_title'] = df_patstat_us_non_cleantech_metadata['appln_title'].apply(' '.join)
df_patstat_us_non_cleantech_metadata['appln_abstract'] = df_patstat_us_non_cleantech_metadata['appln_abstract'].apply(' '.join)
df_patstat_us_cleantech_metadata.reset_index(drop=True, inplace=True)
df_patstat_us_non_cleantech_metadata.reset_index(drop=True, inplace=True)

df_patstat_us_cleantech_metadata['embedding'] = model.encode((df_patstat_us_cleantech_metadata['appln_title'] + ' [SEP] ' + df_patstat_us_cleantech_metadata['appln_abstract']), show_progress_bar=True).tolist()
df_patstat_us_non_cleantech_metadata['embedding'] = model.encode((df_patstat_us_non_cleantech_metadata['appln_title'] + ' [SEP] ' + df_patstat_us_non_cleantech_metadata['appln_abstract']), show_progress_bar=True).tolist()

df_patstat_cleantech_citations_en.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_cleantech_citations_en_grouped_patentsberta.csv')
df_patstat_non_cleantech_citations_en.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_non_cleantech_citations_en_grouped_patentsberta.csv')

df_patstat_us_cleantech_metadata.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_us_cleantech_metadata_patentsberta.csv')
df_patstat_us_non_cleantech_metadata.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_patstat_us_non_cleantech_metadata_patentsberta.csv')

df_patstat_cleantech_citations = pd.merge(df_patstat_cleantech_citations, df_patstat_publn_id_cleantech, on="pat_publn_id", how="inner")
df_patstat_non_cleantech_citations = pd.merge(df_patstat_non_cleantech_citations, df_patstat_publn_id_non_cleantech, on="pat_publn_id", how="inner")


# Function to compute cosine similarity for a chunk of data
def compute_cosine_similarity(chunk, embedding_dict_metadata, embedding_dict_citations):
    cosine_similarity_list = []
    
    # Calculate cosine similarity for each pair of application ID and cited application ID
    for index, row in chunk.iterrows():
        appln_id = row['appln_id']
        cited_appln_id = row['cited_appln_id']

        embedding_appln_id = embedding_dict_metadata.get(appln_id)
        embedding_cited_appln_id = embedding_dict_citations.get(cited_appln_id)

        if embedding_appln_id is not None and embedding_cited_appln_id is not None:
            embedding_appln_id = np.array(embedding_appln_id).reshape(1, -1)
            embedding_cited_appln_id = np.array(embedding_cited_appln_id).reshape(1, -1)

            cosine_sim = cosine_similarity(
                embedding_appln_id,
                embedding_cited_appln_id,
            )[0][0]
        else:
            cosine_sim = np.nan
        
        cosine_similarity_list.append((appln_id, cited_appln_id, cosine_sim))
    
    return cosine_similarity_list

embedding_dict_metadata_cleantech = pd.Series(
    df_patstat_us_cleantech_metadata.embedding.values,
    index=df_patstat_us_cleantech_metadata.appln_id,
).to_dict()

embedding_dict_citations_cleantech = pd.Series(
    df_patstat_cleantech_citations_en.embedding.values,
    index=df_patstat_cleantech_citations_en.cited_appln_id,
).to_dict()

cpu_cores = min(mp.cpu_count(), 18)

# Split the dataset into chunks for parallel processing
data_chunks = np.array_split(df_patstat_cleantech_citations, cpu_cores)

with mp.Pool(cpu_cores) as pool:
    # Process the data chunks in parallel
    results = pool.starmap(
        compute_cosine_similarity,
        [(chunk, embedding_dict_metadata_cleantech, embedding_dict_citations_cleantech) for chunk in data_chunks]
    )

cosine_similarity_list = [item for sublist in results for item in sublist]

df_cosine_similarity_cleantech = pd.DataFrame(
    cosine_similarity_list,
    columns=["appln_id", "cited_appln_id", "cosine_similarity"]
)

df_cosine_similarity_cleantech["cosine_similarity"].fillna(0, inplace=True)

embedding_dict_metadata_non_cleantech = pd.Series(
    df_patstat_us_non_cleantech_metadata.embedding.values,
    index=df_patstat_us_non_cleantech_metadata.appln_id,
).to_dict()

embedding_dict_citations_non_cleantech = pd.Series(
    df_patstat_non_cleantech_citations_en.embedding.values,
    index=df_patstat_non_cleantech_citations_en.cited_appln_id,
).to_dict()

cpu_cores = min(mp.cpu_count(), 18)

# Split the dataset into chunks for parallel processing
data_chunks = np.array_split(df_patstat_non_cleantech_citations, cpu_cores)

with mp.Pool(cpu_cores) as pool:
    # Process the data chunks in parallel
    results = pool.starmap(
        compute_cosine_similarity,
        [(chunk, embedding_dict_metadata_non_cleantech, embedding_dict_citations_non_cleantech) for chunk in data_chunks]
    )

cosine_similarity_list = [item for sublist in results for item in sublist]

df_cosine_similarity_non_cleantech = pd.DataFrame(
    cosine_similarity_list,
    columns=["appln_id", "cited_appln_id", "cosine_similarity"]
)

df_cosine_similarity_non_cleantech["cosine_similarity"].fillna(0, inplace=True)

df_cosine_similarity_cleantech.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_cosine_similarity_cleantech.csv')
df_cosine_similarity_non_cleantech.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_cosine_similarity_non_cleantech.csv')


df_cosine_similarity_grouped_cleantech = df_cosine_similarity_cleantech.groupby('appln_id').agg({'cosine_similarity': ['mean', 'std']}).reset_index()
df_cosine_similarity_grouped_cleantech.columns = ['appln_id', 'cosine_similarity_mean', 'cosine_similarity_std']

df_cosine_similarity_grouped_non_cleantech = df_cosine_similarity_non_cleantech.groupby('appln_id').agg({'cosine_similarity': ['mean', 'std']}).reset_index()
df_cosine_similarity_grouped_non_cleantech.columns = ['appln_id', 'cosine_similarity_mean', 'cosine_similarity_std']

df_cosine_similarity_grouped_cleantech.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_cosine_similarity_grouped_cleantech.csv')
df_cosine_similarity_grouped_non_cleantech.to_csv('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/df_cosine_similarity_grouped_non_cleantech.csv')