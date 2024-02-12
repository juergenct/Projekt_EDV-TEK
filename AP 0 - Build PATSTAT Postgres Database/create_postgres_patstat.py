import os
import csv
import psycopg2
from tqdm import tqdm

# Connect to Postgres server
db_params = {
    'user': 'tie',
    'password': 'TIE%2023!tuhh',
    'host': '127.0.0.1',
    'port': 65432,
    'database': 'PATSTAT_2023'
    # 'database': db_name
}

conn = psycopg2.connect(**db_params)
conn.autocommit = True

# Create new database
# with conn.cursor() as cur:
    # cur.execute(f'CREATE DATABASE {db_name};')
# conn.close()

# Create list of all folders
folders = [
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_01',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_02',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_03',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_04',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_05',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_06',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_07',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_08',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_09',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_10',
    '/mnt/hdd01/PATSTAT_Spring_2023/data_PATSTAT_Global_2023_Spring_11'
]

# Create list of all table names - from Patstat Data Catalog
table_names = [
    'tls201_appln',
    'tls202_appln_title',
    'tls203_appln_abstr',
    'tls204_appln_prior',
    'tls205_tech_rel',
    'tls206_person',
    'tls207_pers_appln',
    'tls209_appln_ipc',
    'tls210_appln_n_cls',
    'tls211_pat_publn',
    'tls212_citation',
    'tls214_npl_publn',
    'tls215_citn_categ',
    'tls216_appln_contn',
    'tls222_appln_jp_class',
    # 'tls223_appln_docus',
    'tls224_appln_cpc',
    'tls225_docdb_fam_cpc',
    'tls226_person_orig',
    'tls227_pers_publn',
    'tls228_docdb_fam_citn',
    'tls229_appln_nace2',
    'tls230_appln_techn_field',
    'tls231_inpadoc_legal_event',
    'tls801_country',
    'tls803_legal_event_code',
    'tls901_techn_field_ipc',
    'tls902_ipc_nace2',
    'tls904_nuts'
]

primary_keys = {
    'tls201_appln': 'appln_id',
    'tls202_appln_title': 'appln_id',
    'tls203_appln_abstr': 'appln_id',
    'tls204_appln_prior': ['appln_id', 'prior_appln_id'],
    'tls205_tech_rel': ['appln_id', 'tech_rel_appln_id'],
    'tls206_person': 'person_id',
    'tls207_pers_appln': ['person_id', 'appln_id', 'applt_seq_nr', 'invt_seq_nr'],
    'tls209_appln_ipc': ['appln_id', 'ipc_class_symbol'],
    'tls210_appln_n_cls': ['appln_id', 'nat_class_symbol'],
    'tls211_pat_publn': 'pat_publn_id',
    'tls212_citation': ['pat_publn_id', 'citn_replenished', 'citn_id'],
    'tls214_npl_publn': 'npl_publn_id',
    'tls215_citn_categ': ['pat_publn', 'citn_replenished', 'citn_id', 'citn_categ', 'relevant_claim'],
    'tls216_appln_contn': ['appln_id', 'parent_appln_id'],
    'tls222_appln_jp_class': ['appln_id', 'jp_class_scheme', 'jp_class_symbol'],
    # 'tls223_appln_docus',
    'tls224_appln_cpc': ['appln_id', 'cpc_class_symbol'],
    'tls225_docdb_fam_cpc': ['docdb_fam_id', 'cpc_class_symbol', 'cpc_gener_auth'],
    'tls226_person_orig': 'person_orig_id',
    'tls227_pers_publn': ['person_id', 'pat_publn_id', 'applt_seq_nr', 'invt_seq_nr'],
    'tls228_docdb_fam_citn': ['docdb_family_id', 'cited_docdb_family_id'],
    'tls229_appln_nace2': ['appln_id', 'nace2_code'],
    'tls230_appln_techn_field': ['appln_id', 'techn_field_nr'],
    'tls231_inpadoc_legal_event': 'event_id',
    'tls801_country': 'ctry_code',
    'tls803_legal_event_code': ['event_auth', 'event_code'],
    'tls901_techn_field_ipc': 'ipc_maingroup_symbol',
    'tls902_ipc_nace2': ['ipc', 'not_with_ipc', 'unless_with_ipc', 'nace2_code'],
    'tls904_nuts': 'nuts'
}

foreign_keys = {
    'tls202_appln_title': [{
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
    }],
    'tls203_appln_abstr': [{
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
    }],
    'tls204_appln_prior': [
        {
            'column': 'appln_id',
            'ref_table': 'tls201_appln',
            'ref_column': 'appln_id'
        },
        {
            'column': 'prior_appln_id',
            'ref_table': 'tls201_appln',
            'ref_column': 'appln_id'
        }
    ],
    'tls205_tech_rel': [
        {
            'column': 'appln_id',
            'ref_table': 'tls201_appln',
            'ref_column': 'appln_id'
        },
        {
            'column': 'tech_rel_appln_id',
            'ref_table': 'tls201_appln',
            'ref_column': 'appln_id'
        }
    ],
    'tls207_pers_appln': [
        {
            'column': 'person_id',
            'ref_table': 'tls206_person',
            'ref_column': 'person_id'
        },
        {
            'column': 'appln_id',
            'ref_table': 'tls201_appln',
            'ref_column': 'appln_id'
        }
    ],
    'tls209_appln_ipc': [{
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
    }],
    'tls210_appln_n_cls': [{
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
    }],
    'tls211_pat_publn': [{
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
    }],
    'tls212_citation': [
        {
            'column': 'pat_publn_id',
            'ref_table': 'tls211_pat_publn',
            'ref_column': 'pat_publn_id'
        },
        {
            'column': 'cited_pat_publn_id',
            'ref_table': 'tls211_pat_publn',
            'ref_column': 'pat_publn_id'
        },
        {
        'column': 'cited_appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
        },
        {
        'column': 'cited_npl_publn_id',
        'ref_table': 'tls214_npl_publn',
        'ref_column': 'npl_publn_id'
    }],
    'tls215_citn_categ': [
        {
        'column': 'pat_publn_id',
        'ref_table': 'tls212_citation',
        'ref_column': 'pat_publn_id'
        },
        {
        'column': 'citn_replenished',
        'ref_table': 'tls212_citation',
        'ref_column': 'citn_replenished'
        },
        {
        'column': 'citn_id',
        'ref_table': 'tls212_citation',
        'ref_column': 'citn_id'
    }],
    'tls216_appln_contn': [
        {
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
        },
        {
        'column': 'parent_appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'

    }],
    'tls222_appln_jp_class': [{
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id'
    }],
    'tls224_appln_cpc': [{
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id',
    }],
    'tls225_docdb_fam_cpc': [{
        'column': 'docdb_family_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'docdb_family_id'
    }],
    'tls226_person_orig': [{
        'column': 'person_id',
        'ref_table': 'tls206_person',
        'ref_column': 'person_id'
    }],
    'tls227_pers_publn': [
        {
        'column': 'person_id',
        'ref_table': 'tls206_person',
        'ref_column': 'person_id'
        },
        {
        'column': 'pat_publn_id',
        'ref_table': 'tls211_pat_publn',
        'ref_column': 'pat_publn_id'
        
    }],
    'tls228_docdb_fam_citn':[
        {
        'column': 'docdb_family_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'docdb_family_id'
        },
        {
        'column': 'cited_docdb_family_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'docdb_family_id'
    }],
    'tls229_appln_nace2': [
        {
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id',
        },
        {
        'column': 'nace2_code',
        'ref_table': 'tls902_ipc_nace2',
        'ref_column': 'nace2_code',
    }],
    'tls230_appln_techn_field': [
        {
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id',
        },
        {
        'column': 'techn_field_nr',
        'ref_table': 'tls901_techn_field_ipc',
        'ref_column': 'techn_field_nr',
    }],
    'tls231_inpadoc_legal_event': [
        {
        'column': 'appln_id',
        'ref_table': 'tls201_appln',
        'ref_column': 'appln_id',
        },
        {
        'column': 'event_auth',
        'ref_table': 'tls803_legal_event_code',
        'ref_column': 'event_auth',
        },
        {
        'column': 'event_code',
        'ref_table': 'tls803_legal_event_code',
        'ref_column': 'event_code',
    }],
    'tls901_techn_field_ipc': [{
        'column': 'techn_field_nr',
        'ref_table': 'tls209_appln_ipc',
        'ref_column': 'techn_field_nr',
    }],
    'tls904_nuts': [{
        'column': 'nuts',
        'ref_table': 'tls206_person',
        'ref_column': 'nuts',
    }]
}


# Create a list of all .csv files and individual .csv files (part01.csv)
individual_csv_files = []
all_csv_files = []

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith('part01.csv') and not file.startswith('._'):
            individual_csv_files.append(os.path.join(folder, file))
        if file.endswith('.csv') and not file.startswith('._'):
            all_csv_files.append(os.path.join(folder, file))

# Function to create tables for each individual .csv file
def create_table_from_csv(cursor, csv_path, table_name):
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

            # Create column definitions
            columns = ', '.join([f"{header} TEXT" for header in headers])
            cursor.execute(f"CREATE TABLE {table_name} ({columns});")
            print(f'Created {table_name} table.')
    except Exception as e:
        print(f"Error creating {table_name} table: {e}")

# Function to insert data into tables for every .csv file (include all parts for each tls table)
def fill_table_from_csv(cursor, csv_path, table_name):
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            copy_query = f"COPY {table_name} FROM STDIN WITH CSV HEADER DELIMITER ',';"
            cursor.copy_expert(copy_query, csvfile)
            print(f'Copied {csv_path} to {table_name} table.')
    except Exception as e:
        print(f"Error copying {csv_path} to {table_name} table: {e}")

# Function to create primary keys or composite primary keys for each table
def create_primary_keys(cursor, primary_keys):
    for table, pk_columns in tqdm(primary_keys.items()):
        # Check if pk_columns is a string or a list and create primary key accordingly
        if isinstance(pk_columns, str):
            pk_columns_str = pk_columns
        elif isinstance(pk_columns, list):
            # Join list elements to create a composite primary key
            pk_columns_str = ', '.join(pk_columns)
        else:
            print(f"Invalid primary key format for table {table}. Must be a string or list.")
            continue

        try:
            cursor.execute(f"ALTER TABLE {table} ADD PRIMARY KEY ({pk_columns_str});")
            print(f"Primary key added to {table} on column(s) {pk_columns_str}")
        except Exception as e:
            print(f"Error adding primary key to {table}: {e}")

def create_foreign_keys(cursor, foreign_keys):
    for table, fk_infos in tqdm(foreign_keys.items()):
        for fk_info in fk_infos:
            fk_column = fk_info['column']
            ref_table = fk_info['ref_table']
            ref_column = fk_info['ref_column']
            constraint_name = f"fk_{table}_{fk_column}_to_{ref_table}_{ref_column}"

            try:
                cursor.execute(f"""
                    ALTER TABLE {table}
                    ADD CONSTRAINT {constraint_name}
                    FOREIGN KEY ({fk_column}) REFERENCES {ref_table}({ref_column});
                """)
                print(f"Foreign key {constraint_name} added to {table} ({fk_column}) referencing {ref_table}({ref_column})")
            except Exception as e:
                print(f"Error adding foreign key to {table}: {e}")

# Create and fill tables
with conn.cursor() as cur:
    # for csv_file in tqdm(individual_csv_files):
    #     table_name = csv_file.split('/')[-1].split('_')[0]
    #     for name in table_names:
    #         if name[3:6] == table_name[3:6]:
    #             table_name = name
    #     create_table_from_csv(cur, csv_file, table_name)
    # for csv_file in tqdm(all_csv_files):
    #     table_name = csv_file.split('/')[-1].split('_')[0]
    #     for name in table_names:
    #         if name[3:6] == table_name[3:6]:
    #             table_name = name
    #     fill_table_from_csv(cur, csv_file, table_name)

    create_primary_keys(cur, primary_keys)

    create_foreign_keys(cur, foreign_keys)

# Close cursor and connection
conn.close()
