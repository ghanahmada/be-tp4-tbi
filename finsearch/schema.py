from typing import List
from pydantic import BaseModel


class Document(BaseModel):
    title: str
    desc: str
    doc_id: str

class SearchRequest(BaseModel):
    query: str
    method: str

class SearchResponse(BaseModel):
    status: int
    data: List[Document]