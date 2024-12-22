from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from finsearch.retrieval.config import ColBERTConfig
from finsearch.retrieval.model.colbert import ColBERTRetriever
from finsearch.service import RetrievalService
from finsearch.schema import SearchRequest, SearchResponse
from fastapi.middleware.cors import CORSMiddleware
from finsearch.util import load_document


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

retrieval_service = RetrievalService(colbert_config=ColBERTConfig)

@app.get("/features")
async def get_features():
    return JSONResponse(content={
        "status": 200,
        "data": retrieval_service.get_features(),
    })

@app.get("/query", response_model=SearchResponse)
async def search(request: SearchRequest):
    result = await retrieval_service.retrieve(request.method, request.query)
    return SearchResponse(status=200, data=result)