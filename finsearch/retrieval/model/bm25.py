# FILE: finsearch/retrieval/model/bm25.py

from typing import List
from finsearch.retrieval.interface import IRetrieval
from finsearch.retrieval.config import BM25Config
from finsearch.util import download_data

class BM25Retriever(IRetrieval):
    def __init__(self, config: BM25Config, collection: List[str]):
        self.config = config
        self.bsbi_instance = BSBIIndex(self.config.bsbi_name, self.config.index_name)
        self.bsbi_instance.load()

    async def retrieve(self, query: str, k: int = 1) -> List[str]:
        results = BSBI_instance.retrieve_bm25_taat(query, 10)
        articles = get_article_mapper(results)
        return articles