from typing import List, Callable, Dict
from fastapi import HTTPException
from finsearch.retrieval.model.colbert import ColBERTRetriever
from finsearch.retrieval.model.bm25 import BM25Retriever
from finsearch.retrieval.config import ColBERTConfig, BM25Config
from finsearch.schema import Document
from finsearch.util import get_article_mapper, load_document, get_docno_mapper


class RetrievalService:
    def __init__(self, colbert_config: ColBERTConfig, bm25_config: BM25Config):
        # document_df = load_document()
        import pandas as pd
        document_df = pd.read_parquet("experiment\data\document.parquet")
        # self.article_to_title = get_article_mapper(document_df)
        # self.collection = document_df["Article"].tolist()
        # self.colbert_searcher = ColBERTRetriever(config=colbert_config, collection=self.collection)
        self.colbert_searcher = None
        
        self.docno_to_articles = get_docno_mapper(document_df)
        self.bm25_collection = document_df["docno"].tolist()
        self.bm25_searcher = BM25Retriever(config=bm25_config, collection=self.bm25_collection)
        
        self.retriever_methods: Dict[str, Callable[[str, int], List[str]]] = {
            # "ColBERT": self.colbert_searcher.retrieve,
            "BM25": self.bm25_searcher.retrieve,  
            "BM25 + Davinci OpenAI": self._bm25_openai_retriever,  
        }

    def get_features(self) -> List[str]:
        return list(self.retriever_methods.keys())

    async def retrieve(self, method: str, query: str, k: int = 30) -> List[Document]:
        retriever = self.retriever_methods.get(method)
        if not retriever:
            raise HTTPException(status_code=400, detail=f"Invalid retrieval method: {method}")
        
        if method == "ColBERT":
            passage_list = await retriever(query, k)
            return [
                Document(title=self.article_to_title[doc]["title"], 
                             desc=doc, 
                             doc_id=self.article_to_title[doc]["docno"]) for doc in passage_list
            ]
        elif method == "BM25":
            docno_list = await retriever(query, k)
            return [
                Document(title=self.docno_to_articles[docno]["title"], 
                             desc=self.docno_to_articles[docno]["desc"], 
                             doc_id=docno) for docno in docno_list
            ]

    async def _bm25_retriever(self, query: str, k: int) -> List[str]:
        raise NotImplementedError("BM25 retriever not implemented yet.")

    async def _bm25_openai_retriever(self, query: str, k: int) -> List[str]:
        raise NotImplementedError("BM25 + OpenAI retriever not implemented yet.")
