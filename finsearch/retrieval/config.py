from dataclasses import dataclass


@dataclass
class ColBERTConfig:
    experiment_path: str = "/app/index/experiments/msmarco"
    index_name: str = "ir"
    index_url: str = "https://drive.google.com/uc?&id=1Y1Eee4mHQ-jDvBUNehFfEmMj-r9iNDcf"
    folder_name: str = "experiments"

@dataclass
class BM25Config:
    experiment_path: str = "/app/index/experiments/bm25"
    index_name: str = "ir"
    bsbi_name: str = "bsbi.py" 
    index_url: str = "https://drive.google.com/drive/folders/1Kv1uvEI508BaIOdhSEwikqJ5ubPkd-DY"
    folder_name: str = "experiments"