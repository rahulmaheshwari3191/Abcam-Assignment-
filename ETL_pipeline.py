import xml.etree.ElementTree as ET
import requests
import pandas as pd
import gzip
import sqlite3
import os


# Helper function for downloading files
def download_file(url, filepath):
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)


# Helper function for reading compressed files
def read_gzip_file(filepath, mode='rt'):
    with gzip.open(filepath, mode) as f:
        return f


# Reusable function for saving DataFrame to CSV
def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)


# Step 1: Extract data from UniProt (XML)
def fetch_uniprot_data(record_limit=1000, xml_filepath="uniprot_sprot.xml.gz"):
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz"
    download_file(url, xml_filepath)

    # Extract and parse XML
    with gzip.open(xml_filepath, "rb") as f:
        tree = ET.parse(f)
        root = tree.getroot()

    namespace = {'uniprot': 'http://uniprot.org/uniprot'}
    data = []

    for entry in root.findall('uniprot:entry', namespace):
        data.append({
            'Primary Accession': entry.find('uniprot:accession', namespace).text,
            'Recommended Protein Name': (entry.find('uniprot:protein/uniprot:recommendedName/uniprot:fullName',
                                                    namespace) or {}).text or '',
            'Species Common Name': (entry.find("uniprot:organism/uniprot:name[@type='common']",
                                               namespace) or {}).text or '',
            'STRING dbReference': (entry.find("uniprot:dbReference[@type='STRING']", namespace) or {}).get('id', ''),
            'OpenTargets dbReference': (entry.find("uniprot:dbReference[@type='OpenTargets']", namespace) or {}).get(
                'id', ''),
            'Sequence Length': (entry.find('uniprot:sequence', namespace) or {}).get('length', ''),
            'Sequence Mass': (entry.find('uniprot:sequence', namespace) or {}).get('mass', '')
        })

    df = pd.DataFrame(data)
    save_to_csv(df, 'uniprot_data.csv')
    return df


# Step 2: Extract data from STRING (TSV)
def fetch_string_data(string_filepath="string_data.tsv.gz"):
    url = "https://string-db.org/cgi/download?sessionId=baXq4yzPPB1H&species_text=Homo+sapiens"
    download_file(url, string_filepath)

    with read_gzip_file(string_filepath) as f:
        string_df = pd.read_csv(f, sep='\t')[['protein1', 'protein2', 'combined_score']]

    save_to_csv(string_df, 'string_data.csv')
    return string_df


# Step 3: Extract data from OpenTargets (Parquet format)
def fetch_opentargets_data(parquet_filepath="targets.parquet"):
    targets_url = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.09/output/etl/parquet/targets/part-00000-b2a23987-7e43-4651-a89c-1eb34ec9f9f5-c000.snappy.parquet"
    download_file(targets_url, parquet_filepath)

    targets_df = pd.read_parquet(parquet_filepath)[['id', 'approvedSymbol', 'biotype']]
    save_to_csv(targets_df, 'targets_data.csv')
    return targets_df


# Step 4: Clean and normalize the data
def clean_and_normalize_data(df, source_name):
    df_clean = (df.dropna()
                .drop_duplicates()
                .applymap(lambda x: x.strip().lower() if isinstance(x, str) else x))

    print(f"Data from {source_name} cleaned and normalized!")
    return df_clean


# Step 5: Create clean data tables in SQLite
def create_table_if_not_exists(conn, table_name, columns):
    cursor = conn.cursor()
    column_definitions = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})")
    conn.commit()


def create_clean_data_tables(conn):
    uniprot_columns = {
        'primary_accession': 'TEXT',
        'recommended_protein_name': 'TEXT',
        'primary_gene_name': 'TEXT',
        'species_common_name': 'TEXT',
        'string_dbReference': 'TEXT',
        'opentargets_dbReference': 'TEXT',
        'sequence_length': 'TEXT',
        'sequence_mass': 'TEXT'
    }
    string_columns = {'protein1': 'TEXT', 'protein2': 'TEXT', 'combined_score': 'REAL'}
    opentargets_columns = {'id': 'TEXT', 'approvedSymbol': 'TEXT', 'biotype': 'TEXT'}
    semantic_columns = {
        'primary_accession': 'TEXT',
        'recommended_protein_name': 'TEXT',
        'primary_gene_name': 'TEXT',
        'species_common_name': 'TEXT',
        'disease': 'TEXT',
        'associated_proteins': 'TEXT'
    }

    create_table_if_not_exists(conn, 'clean_uniprot', uniprot_columns)
    create_table_if_not_exists(conn, 'clean_string', string_columns)
    create_table_if_not_exists(conn, 'clean_targets', opentargets_columns)
    create_table_if_not_exists(conn, 'semantic_layer', semantic_columns)

    print("Clean data tables created successfully!")


# Step 6: Insert cleaned data into the clean tables
def insert_cleaned_data(df, table_name, conn):
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Cleaned data inserted into '{table_name}' successfully!")


# Step 7: Create the Semantic Data Layer
def create_semantic_layer(conn):
    uniprot_df = pd.read_sql('SELECT * FROM clean_uniprot', conn)
    string_df = pd.read_sql('SELECT * FROM clean_string', conn)
    opentargets_df = pd.read_sql('SELECT * FROM clean_targets', conn)

    semantic_layer_df = (
        uniprot_df
        .merge(string_df[string_df['combined_score'] > 200], left_on='string_dbReference', right_on='protein1',
               how='left')
        .merge(opentargets_df, left_on='opentargets_dbReference', right_on='id', how='left')
        .groupby(['primary_accession', 'recommended_protein_name', 'primary_gene_name', 'species_common_name'])
        .agg(
            disease=('approvedSymbol', lambda x: ', '.join(x.dropna().unique())),
            associated_proteins=('protein2', lambda x: ', '.join(x.dropna().unique()))
        )
        .reset_index()
    )

    insert_cleaned_data(semantic_layer_df, 'semantic_layer', conn)
    print("Semantic Layer table created successfully!")


# Step 8: Main function to fetch, clean, and insert cleaned data
def main():
    conn = sqlite3.connect('etl_pipeline.db')

    # Fetch raw data
    uniprot_df = fetch_uniprot_data()
    string_df = fetch_string_data()
    opentargets_df = fetch_opentargets_data()

    # Clean and normalize data
    clean_uniprot_df = clean_and_normalize_data(uniprot_df, 'UniProt')
    clean_string_df = clean_and_normalize_data(string_df, 'STRING')
    clean_opentargets_df = clean_and_normalize_data(opentargets_df, 'OpenTargets')

    # Create clean data tables
    create_clean_data_tables(conn)

    # Insert cleaned data into clean tables
    insert_cleaned_data(clean_uniprot_df, 'clean_uniprot', conn)
    insert_cleaned_data(clean_string_df, 'clean_string', conn)
    insert_cleaned_data(clean_opentargets_df, 'clean_targets', conn)

    # Create Semantic Layer
    create_semantic_layer(conn)

    conn.close()


if __name__ == "__main__":
    main()
