import torch
import re
import pandas as pd
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('anferico/bert-for-patents').to(device)

g_patent = pd.read_csv('/mnt/hdd01/patentsview/Raw files/Raw zip files/g_patent.tsv.zip', sep='\t', compression='zip', usecols=['patent_id', 'patent_title', 'patent_abstract'], low_memory=False)
g_cpc = pd.read_csv('/mnt/hdd01/patentsview/Raw files/Raw zip files/g_cpc_current.tsv.zip', sep='\t', compression='zip', usecols=['patent_id', 'cpc_class'], low_memory=False)
g_patent = g_patent.astype(str)
g_cpc = g_cpc.astype(str)

g_patent_cpc = pd.merge(g_patent, g_cpc, on='patent_id')
g_patent_cpc = g_patent_cpc.groupby('patent_id').agg({
    'cpc_class': list,
    'patent_title': 'first',
    'patent_abstract': 'first'
}).reset_index().rename(columns={'cpc_class': 'cpc_groups'})

g_patent_cleantech = g_patent_cpc[g_patent_cpc['cpc_groups'].apply(lambda x: 'Y02' in x)]
g_patent_non_cleantech = g_patent_cpc.sample(n=len(g_patent_cleantech), random_state=42)

def preprocess_text(text):
    # text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

g_patent_cleantech.loc[:, 'patent_title'] = g_patent_cleantech['patent_title'].apply(preprocess_text)
g_patent_cleantech.loc[:, 'patent_abstract'] = g_patent_cleantech['patent_abstract'].apply(preprocess_text)
g_patent_non_cleantech.loc[:, 'patent_title'] = g_patent_non_cleantech['patent_title'].apply(preprocess_text)
g_patent_non_cleantech.loc[:, 'patent_abstract'] = g_patent_non_cleantech['patent_abstract'].apply(preprocess_text)

g_patent_cleantech.loc[:, 'patent_title_abstract'] = g_patent_cleantech['patent_title'] + ' [SEP] ' + g_patent_cleantech['patent_abstract']
g_patent_non_cleantech.loc[:, 'patent_title_abstract'] = g_patent_non_cleantech['patent_title'] + ' [SEP] ' + g_patent_non_cleantech['patent_abstract']

g_patent_cleantech.loc[:, 'label'] = 1
g_patent_non_cleantech.loc[:, 'label'] = 0

g_patent = pd.concat([g_patent_cleantech, g_patent_non_cleantech], ignore_index=True)

g_patent = g_patent.sort_values(by=['patent_id']).reset_index(drop=True)

g_patent['index'] = g_patent.index

g_patent['patent_title_abstract_bert_for_patents_embedding'] = model.encode(g_patent['patent_title_abstract'].tolist(), show_progress_bar=True, device=device).tolist()

g_patent.to_csv('/mnt/hdd01/patentsview/Graph Neural Network for EDV-TEK/g_patent_embedding_test.csv')