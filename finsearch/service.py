from collections import namedtuple
from typing import List, Callable, Dict
from fastapi import HTTPException
from finsearch.retrieval.model.colbert import ColBERTRetriever
from finsearch.retrieval.model.bm25 import BM25Retriever, BM25RetrieverOpenAI, BM25RetrieverTFIDF
from finsearch.retrieval.config import ColBERTConfig, BM25Config
from finsearch.schema import Document
from finsearch.util import get_article_mapper, load_document, get_docno_mapper


class RetrievalService:
    def __init__(self, colbert_config: ColBERTConfig, bm25_config: BM25Config):
        document_df = load_document()
        self.article_to_title = get_article_mapper(document_df)
        self.docno_to_articles = get_docno_mapper(document_df)
        self.collection = document_df["Article"].tolist()
        self.bm25_collection = document_df["docno"].tolist()
        self.RetrievalConfig = namedtuple("RetrievalConfig", ["method", "mapper"])

        colbert_searcher = ColBERTRetriever(config=colbert_config, collection=document_df["Article"].tolist())
        bm25_searcher = BM25Retriever(config=bm25_config, collection=document_df["docno"].tolist())
        bm25_openai_searcher = BM25RetrieverOpenAI(base=bm25_searcher, config=bm25_config, mapper=self.docno_to_articles)
        bm25_tfidf_searcher = BM25RetrieverTFIDF(base=bm25_searcher, config=bm25_config, mapper=self.docno_to_articles)

        self.retriever_methods: Dict[str, self.RetrievalConfig] = {
            "ColBERT": self.RetrievalConfig(method=colbert_searcher.retrieve, mapper=self.article_to_title),
            "BM25": self.RetrievalConfig(method=bm25_searcher.retrieve, mapper=self.docno_to_articles),
            "BM25 with OpenAI": self.RetrievalConfig(method=bm25_openai_searcher.retrieve, mapper=self.docno_to_articles),
            "BM25 with TFIDF": self.RetrievalConfig(method=bm25_tfidf_searcher.retrieve, mapper=self.docno_to_articles),
        }

    def get_features(self) -> List[str]:
        return list(self.retriever_methods.keys())

    async def retrieve(self, method: str, query: str, k: int = 30) -> List[Document]:
        retrieval_config = self.retriever_methods.get(method)
        if not retrieval_config:
            raise HTTPException(status_code=400, detail=f"Invalid retrieval method: {method}")

        result_list = await retrieval_config.method(query, k)
        mapper = retrieval_config.mapper

        return [
            Document(
                title=mapper[result]["title"],
                desc=mapper[result].get("desc", result),  
                doc_id=mapper[result]["docno"]
            )
            for result in result_list
        ]
