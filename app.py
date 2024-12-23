from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
from finsearch.retrieval.config import ColBERTConfig
from finsearch.service import RetrievalService
from finsearch.schema import SearchResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

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
async def search(query: str = Query(...), method: str = Query(...)):
    result = await retrieval_service.retrieve(method=method, query=query)
    return SearchResponse(status=200, data=result)