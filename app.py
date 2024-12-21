import asyncio
from fastapi import FastAPI, HTTPException

from finsearch.api import HealthCheckResponse


app = FastAPI()

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="running",
    )
