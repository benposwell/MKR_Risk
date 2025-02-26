from supabase import create_client
import os
import pandas as pd
from io import BytesIO
import streamlit as st

@st.cache_resource
def connect_to_supabase(): 
    URL = st.secrets['SUPABASE_URL']
    KEY = st.secrets['SUPABASE_KEY']
    supabase = create_client(URL, KEY)
    return supabase

def list_csv_files(supabase, bucket_name):
    """Retrieve a list of all .csv files in the Supabase storage bucket."""
    response = supabase.storage.from_(bucket_name).list()
    
    if 'error' in response and response['error']:
        print("Error fetching file list:", response['error']['message'])
        return []
    
    return [file['name'] for file in response if file['name'].endswith('.csv')]

def load_csv_to_dataframe(supabase, bucket_name, file_name):
    """Load a CSV file from Supabase Storage into a Pandas DataFrame."""
    response = supabase.storage.from_(bucket_name).download(file_name)

    if response:
        df = pd.read_csv(BytesIO(response))
        return df
    else:
        print(f"Failed to load: {file_name}")
        return None
    
@st.cache_data
def load_all_csvs_to_dataframe(bucket_name):
    """Load all CSV files from a Supabase bucket into a list of Pandas DataFrames."""
    supabase = connect_to_supabase()
    csv_files = list_csv_files(supabase, bucket_name)
    
    if not csv_files:
        st.warning("No CSV files found in the bucket.")
        return None
    
    # st.write(f"Found {len(csv_files)} CSV files. Loading into DataFrames...")
    dataframes = {file_name: load_csv_to_dataframe(supabase, bucket_name, file_name) for file_name in csv_files}
    
    return dataframes  # Returns a dictionary of {filename: dataframe}


# # temporary function load_all_csvs_to_dataframe to load all csvs from the 'data' directory. remove 'data' from the file name when saving
# @st.cache_data
# def load_all_csvs_to_dataframe(bucket_name):
#     dataframes = {}
#     for file in os.listdir('data'):
#         if file.endswith('.csv'):
#             dataframes[file.replace('data/', '')] = pd.read_csv(os.path.join('data', file))
#     return dataframes