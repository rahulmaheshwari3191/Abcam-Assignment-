import unittest
import pandas as pd
import sqlite3
import requests
import gzip
from unittest import mock
from ETL_pipeline import (
    fetch_uniprot_data, fetch_string_data, fetch_opentargets_data,
    clean_and_normalize_data, create_clean_data_tables, insert_cleaned_data,
    create_semantic_layer, main  # Import main function here
)


# Test fetching UniProt data
@mock.patch('ETL_pipeline.requests.get')  # Correct patch path
def test_fetch_uniprot_data(mock_get):
    # Simulate gzipped XML content
    mock_get.return_value.content = gzip.compress(b"""<?xml version="1.0"?>
    <uniprot>
        <entry>
            <accession>P12345</accession>
            <protein>
                <recommendedName><fullName>Protein1</fullName></recommendedName>
            </protein>
            <organism>
                <name type="common">Human</name>
            </organism>
            <dbReference type="STRING" id="9606.ENSP00000354587"/>
            <dbReference type="OpenTargets" id="ENSG00000141510"/>
        </entry>
    </uniprot>""")

    df = fetch_uniprot_data(record_limit=1)
    assert not df.empty
    assert 'Primary Accession' in df.columns


# Test fetching STRING data
@mock.patch('ETL_pipeline.requests.get')  # Correct patch path
def test_fetch_string_data(mock_get):
    # Simulate gzipped TSV content
    mock_get.return_value.content = gzip.compress(b"protein1\tprotein2\tcombined_score\n"
                                                  b"9606.ENSP00000354587\t9606.ENSP00000354588\t900\n")
    df = fetch_string_data()
    assert not df.empty
    assert 'protein1' in df.columns
    assert 'protein2' in df.columns


# Test fetching OpenTargets data
@mock.patch('ETL_pipeline.requests.get')  # Correct patch path
def test_fetch_opentargets_data(mock_get):
    # Simulate parquet data
    parquet_data = pd.DataFrame({
        'id': ['ENSG00000141510'],
        'approvedSymbol': ['BRCA1'],
        'biotype': ['protein_coding']
    }).to_parquet()

    mock_get.return_value.content = parquet_data
    df = fetch_opentargets_data()
    assert not df.empty
    assert 'id' in df.columns


# Test cleaning and normalizing data
def test_clean_and_normalize_data():
    df = pd.DataFrame({
        'col1': [' Value1 ', None, 'value1'],
        'col2': [' DUPLICATE ', 'duplicate', None]
    })
    cleaned_df = clean_and_normalize_data(df, 'test_source')
    assert cleaned_df.shape[0] == 1
    assert cleaned_df['col1'].iloc[0] == 'value1'


# Test SQLite table creation
def test_create_clean_data_tables():
    conn = sqlite3.connect(':memory:')  # In-memory SQLite database
    create_clean_data_tables(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    assert ('clean_uniprot',) in tables


# Test inserting cleaned data into SQLite
def test_insert_cleaned_data():
    conn = sqlite3.connect(':memory:')
    df = pd.DataFrame({
        'col1': ['value1', 'value2'],
        'col2': [10, 20]
    })
    insert_cleaned_data(df, 'test_table', conn)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test_table")
    rows = cursor.fetchall()
    assert len(rows) == 2


# Test semantic layer creation
def test_create_semantic_layer():
    conn = sqlite3.connect(':memory:')
    # Mock data in the database
    uniprot_df = pd.DataFrame({
        'primary_accession': ['P12345'],
        'recommended_protein_name': ['Protein1'],
        'string_dbReference': ['9606.ENSP00000354587']
    })
    uniprot_df.to_sql('clean_uniprot', conn, index=False)

    string_df = pd.DataFrame({
        'protein1': ['9606.ENSP00000354587'],
        'protein2': ['9606.ENSP00000354588'],
        'combined_score': [900]
    })
    string_df.to_sql('clean_string', conn, index=False)

    opentargets_df = pd.DataFrame({
        'id': ['ENSG00000141510'],
        'approvedSymbol': ['BRCA1']
    })
    opentargets_df.to_sql('clean_targets', conn, index=False)

    create_semantic_layer(conn)

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM semantic_layer")
    rows = cursor.fetchall()
    assert len(rows) > 0


# Test main flow
@mock.patch('ETL_pipeline.requests.get')  # Correct patch path
def test_main(mock_get):
    # Simulate request content
    mock_get.return_value.content = b'fake data'
    conn = sqlite3.connect(':memory:')  # Use in-memory database for testing

    # Call the main function to test the entire ETL process
    main()

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    assert ('clean_uniprot',) in tables
    assert ('semantic_layer',) in tables