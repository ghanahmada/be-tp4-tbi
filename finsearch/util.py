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

def load_document():  
    return pd.read_parquet("/experiment/data/document.parquet")

def get_article_mapper(document_df: pd.DataFrame):
    article_to_title = {
            dct["Article"]: {
                "title": dct["Article_title"], 
                "docno": str(dct["docno"])
            } for dct in document_df.to_dict(orient="records")
        }
    return article_to_title
