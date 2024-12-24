import os
import gdown
import zipfile
import logging
import pandas as pd
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_data(url, filename, dir_name: str = "index") -> None:
    original_dir = os.getcwd()
    target_dir = os.path.join(original_dir, dir_name)
    
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    file_path = os.path.join(target_dir, filename)

    logging.info("Downloading data....")
    gdown.download(url, output=file_path, quiet=False)
    logging.info(f"Downloaded to: {file_path}")

    if file_path.endswith(".zip"):
        logging.info("Extracting zip file....")

        extract_folder = os.path.splitext(file_path)[0]
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        logging.info(f"Extracted files to: {extract_folder}")

        os.remove(file_path)
        logging.info("Removed zip file after extraction.")
    else:
        logging.info("No ZIP extension detected. Skipping extraction.")

def load_document():  
    return pd.read_parquet("/app/experiment/data/document.parquet")

def get_article_mapper(document_df: pd.DataFrame):
    article_to_title = {
            dct["Article"]: {
                "title": dct["Article_title"], 
                "docno": str(dct["docno"])
            } for dct in document_df.to_dict(orient="records")
        }
    return article_to_title

def get_docno_mapper(document_df: pd.DataFrame):
    import pandas as pd
    dict_docs = {str(dct['docno']): {
                    "title":dct['Article_title'],
                    "desc":dct['Article'],
                    "doc_id":dct["docno"]
                } for dct in document_df.to_dict(orient='records')}
    return dict_docs