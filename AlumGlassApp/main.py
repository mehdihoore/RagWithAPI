from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from functools import lru_cache
import logging

from services.astra_service import AstraService
from services.openai_service import OpenAIService
from services.search_services import (
    GoogleSearchService,
    DuckDuckGoService,
    WikipediaService,
    StackExchangeService,
    SearxNGService,
)
from config import Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

config = Config()
astra_service = AstraService(config)
openai_service = OpenAIService(config)

google_service = GoogleSearchService(config)
ddg_service = DuckDuckGoService()
wiki_service = WikipediaService()
stack_service = StackExchangeService()
searxng_service = SearxNGService(config)


class QueryRequest(BaseModel):
    query: str
    collection_name: str = Field(default="mabahes")
    prompt: Optional[str] = None
    print_results: bool = Field(default=True)
    model: Optional[str] = Field(default="gpt-4.0-turbo")


class QueryResponse(BaseModel):
    answer: str
    mabhas_references: List[str]
    confidence: str
    additional_sources: List[str]


@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    return openai_service.get_embedding(text)


def hybrid_search(query_text: str, collection_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    query_embedding = get_embedding(query_text)
    return astra_service.hybrid_search(query_text, query_embedding, collection_name, top_k)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"An error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )


@app.post("/query", response_model=QueryResponse)
async def query_alumglass(request: QueryRequest):
    try:
        logging.info(f"Received query: {request.query}")

        # Perform searches
        search_results = hybrid_search(request.query, request.collection_name)
        if request.print_results:
            print("Search Results from Astra Service:", search_results)

        google_results = google_service.search(request.query)
        if request.print_results:
            print("Google Search Results:", google_results)

        ddg_result = ddg_service.search(request.query)
        if request.print_results:
            print("DuckDuckGo Search Results:", ddg_result)

        wiki_result = wiki_service.search(request.query)
        if request.print_results:
            print("Wikipedia Search Results:", wiki_result)

        stack_results = stack_service.search(request.query)
        if request.print_results:
            print("Stack Exchange Search Results:", stack_results)

        searxng_result = searxng_service.search(request.query)
        if request.print_results:
            print("SearxNG Search Results:", searxng_result)

        # Use the specified model for OpenAI service
        openai_answer = openai_service.get_response(
            request.query,
            search_results,
            google_results,
            ddg_result,
            wiki_result,
            stack_results,
            searxng_result,
            custom_prompt=request.prompt,
            model=request.model  # Pass the model parameter
        )

        # Ensure the mabhas_references and additional_sources are lists
        if isinstance(openai_answer['mabhas_references'], str):
            openai_answer['mabhas_references'] = [
                openai_answer['mabhas_references']]
        if isinstance(openai_answer['additional_sources'], str):
            openai_answer['additional_sources'] = [
                openai_answer['additional_sources']]

        return QueryResponse(**openai_answer)
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
