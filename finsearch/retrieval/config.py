from dataclasses import dataclass


@dataclass
class ColBERTConfig:
    experiment_path: str = "/app/index/roberta_experiments/msmarco"
    index_name: str = "ir"
    index_url: str = "https://drive.google.com/uc?&id=1rKzsF9BWinFJGKr3nqCp26-gK1GQxzVE"
    folder_name: str = "roberta_experiments.zip"

@dataclass
class BM25Config:
    output_dir: str = "/app/index/bm25_index"
    data_dir: str = r'\app\index\arxiv_collections'
    embed_openai_path: str = "/app/index/document_embedded.json"
    tfidf_model: str = "/app/finsearch/retrieval/model/constant/model_tfidf.json"
    openai_model: str = "/app/finsearch/retrieval/model/constant/model_openai.json"

    arxiv_collections_folder: str = "arxiv_collections.zip"
    index_folder: str = "bm25_index.zip"
    embed_filename: str = "document_embedded.json"

    index_url: str = "https://drive.google.com/uc?&id=1PDsalRBWyFL4lm5a7zGUpXmGbtkcO9DI"
    embed_url: str = "https://drive.google.com/uc?&id=1Dbrbs1bHfbcVh9PRtxmJPQDDg9ypZTQw"
    arxiv_collections_url: str = "https://drive.google.com/uc?&id=18o8MduiMdkBOpa--6Nere7uGQNxJohaT"