from typing import List
from pydantic import BaseModel


class Document(BaseModel):
    title: str
    desc: str
    doc_id: str
    
class SearchResponse(BaseModel):
    status: int
    data: List[Document]