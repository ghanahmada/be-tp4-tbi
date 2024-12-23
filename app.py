import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from finsearch.api import HealthCheckResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="running",
    )

@app.get("/features")
def get_features():
    return JSONResponse(content={
        "status": 200,
        "data": [
            "BM25",
            "ColBERT",
            "BM25 + Davinci OpenAI"
        ]
    })
