import os
from typing import Optional, Annotated, List, Optional
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Header
from datura.dataset.tool_return import ResponseOrder
from datura.dataset.date_filters import DateFilterType
from datura.protocol import Model
from datura.utils import get_max_execution_time
import uvicorn
import bittensor as bt
import traceback
from validator import Neuron
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

EXPECTED_ACCESS_KEY = os.environ.get("EXPECTED_ACCESS_KEY", "test")

neu = Neuron()


available_tools = [
    "Twitter Search",
    "Google Search",
    "Google News Search",
    "Google Image Search",
    "Bing Search",
    "ArXiv Search",
    "Wikipedia Search",
    "Youtube Search",
    "Hacker News Search",
    "Reddit Search",
]


def format_enum_values(enum):
    values = [value.value for value in enum]
    values = ", ".join(values)

    return f"Options: {values}"


class SearchRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Search query prompt",
        example="What are the recent sport events?",
    )
    tools: List[str] = Field(
        ..., description="List of tools to search with", example=available_tools
    )
    response_order: Optional[ResponseOrder] = Field(
        default=ResponseOrder.LINKS_FIRST,
        description=f"Order of the search results. {format_enum_values(ResponseOrder)}",
    )
    date_filter: Optional[DateFilterType] = Field(
        default=DateFilterType.PAST_WEEK,
        description=f"Date filter for the search results.{format_enum_values(DateFilterType)}",
        example=DateFilterType.PAST_WEEK.value,
    )
    model: Optional[Model] = Field(
        default=Model.NOVA,
        description=f"Model to use for scraping. {format_enum_values(Model)}",
        example=Model.NOVA.value,
    )


fields = "\n".join(
    f"- {key}: {item.get('description')}"
    for key, item in SearchRequest.schema().get("properties", {}).items()
)

SEARCH_DESCRIPTION = f"""Performs a search across multiple platforms. Available tools are:
- Twitter Search: Uses Twitter API to search for tweets in past week date range.
- Google Search: Searches the web using Google.
- Google News Search: Searches news articles using Google News.
- Google Image Search: Searches images using Google.
- Bing Search: Searches the web using Bing.
- ArXiv Search: Searches academic papers on ArXiv.
- Wikipedia Search: Searches articles on Wikipedia.
- Youtube Search: Searches videos on Youtube.
- Hacker News Search: Searches posts on Hacker News, under the hood it uses Google search.
- Reddit Search: Searches posts on Reddit, under the hood it uses Google search.

Request Body Fields:
{fields}
"""



async def response_stream_event(data: SearchRequest):
    try:
        query = {
            "content": data.prompt,
            "tools": data.tools,
            "date_filter": data.date_filter.value,
            "response_order": data.response_order,
        }

        max_execution_time = get_max_execution_time(data.model)

        merged_chunks = ""

        async for response in neu.scraper_validator.organic(query, max_execution_time):
            # Decode the chunk if necessary and merge
            chunk = str(response)  # Assuming response is already a string
            merged_chunks += chunk
            lines = chunk.split("\n")
            sse_data = "\n".join(f"data: {line if line else ' '}" for line in lines)
            yield f"{sse_data}\n\n"
    except Exception as e:
        bt.logging.error(f"error in response_stream {traceback.format_exc()}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post(
    "/search",
    summary="Search across multiple platforms",
    description=SEARCH_DESCRIPTION,
    response_description="A stream of search results from the specified tools.",
)
async def search(
    body: SearchRequest, access_key: Annotated[str | None, Header()] = None
):
    """
    Search endpoint that accepts a JSON body with search parameters.
    """

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return StreamingResponse(response_stream_event(body))


@app.get("/")
async def health_check():
    return {"status": "healthy"}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Datura API",
        version="1.0.0",
        summary="API for searching across multiple platforms",
        routes=app.routes,
        servers=[
            {"url": "https://api.smartscrape.ai", "description": "Datura API"},
            {"url": "http://localhost:8005", "description": "Datura API"},
        ],
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8005, timeout_keep_alive=300)


if __name__ == "__main__":
    asyncio.get_event_loop().create_task(neu.run())
    run_fastapi()
