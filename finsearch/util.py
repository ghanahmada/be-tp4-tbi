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
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    logging.info("Downloading data....")
    gdown.download(url, quiet=False)
    logging.info(f"Downloading to: {os.getcwd()}")

    logging.info("Extracting zip file....")
    with zipfile.ZipFile(f"{filename}.zip", 'r') as zip_ref:
        zip_ref.extractall(filename)
    logging.info(f"Extracted files to: {filename}")

    os.remove(f"{filename}.zip")
    logging.info("Removed zip file after extraction.")

    os.chdir("..")

def download_data_bm25(url, filename, dir_name: str = "index") -> None:
    # Pastikan direktori target ada
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    if os.path.isdir(dir_name) and os.listdir(dir_name):
        logging.info(f"Directory '{dir_name}' already exists and is not empty. Skipping download.")
        return

    logging.info("Downloading data....")
    gdown.download(url, quiet=False)
    logging.info(f"Downloaded to: {os.getcwd()}")

    logging.info("Extracting zip file....")
    with zipfile.ZipFile(f"{filename}.zip", 'r') as zip_ref:
        zip_ref.extractall(dir_name)  # Ekstrak langsung ke dir_name
    logging.info(f"Extracted files to: {dir_name}")

    os.remove(f"{filename}.zip")
    logging.info("Removed zip file after extraction.")

def download_data_openai_embedder(url, filename, dir_name) -> None:
    # Pastikan direktori target ada
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    output_path = os.path.join(dir_name, filename)
    
    # Cek apakah file sudah ada
    if os.path.isfile(output_path):
        logging.info(f"File '{output_path}' already exists. Skipping download.")
        return

    logging.info("Downloading data....")
    gdown.download(url, output=output_path, quiet=False)
    logging.info(f"Downloaded to: {output_path}")

def load_document():  
    # return pd.read_parquet("/app/experiment/data/document.parquet")
    return pd.read_parquet("experiment/data/document.parquet")

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