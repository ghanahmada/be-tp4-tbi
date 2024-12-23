from typing import Any, List
from colbert import Searcher
from colbert.infra import Run, RunConfig
from finsearch.retrieval.interface import IRetrieval
from finsearch.retrieval.config import ColBERTConfig
from finsearch.util import download_data


class ColBERTRetriever(IRetrieval):
    def __init__(self, config: ColBERTConfig, collection: List[str]):
        self.config = config
        self.collection = collection
        self.setup_index()

    def setup_index(self):
        download_data(url=self.config.index_url, filename=self.config.folder_name)
        
        with Run().context(RunConfig(experiment=self.config.experiment_path)):
            self.searcher = Searcher(index=self.config.index_name, collection=self.collection)
            print(len(self.searcher.collection))

        self.searcher.search("init...", k=1)

    async def retrieve(self, query: str, k: int=1):
        passage_id, passage_rank, passage_score = self.searcher.search(query, k=k)
        passage_retrieved = [self.collection[i] for i in passage_id]
        return passage_retrieved