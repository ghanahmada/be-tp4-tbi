from abc import ABC, abstractmethod
from typing import List
from finsearch.schema import Document


class IRetrieval(ABC):
    @abstractmethod
    def setup_index(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    async def retrieve(self, query: str) -> List[str]:
        raise NotImplementedError