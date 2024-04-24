import os
from pydantic import BaseModel, Field
from typing import List
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request, Query
import uvicorn
import bittensor as bt
import traceback
from validator import Neuron
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import json
# from datura.tools.search.serp_google_search_tool import SerpGoogleSearchTool
# from datura.tools.search.serp_google_image_search_tool import SerpGoogleImageSearchTool
# from datura.tools.hacker_news.hacker_news_search_tool import HackerNewsSearchTool
# from datura.tools.reddit.reddit_search_tool import RedditSearchTool

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
EXPECTED_ACCESS_KEY = os.environ.get("EXPECTED_ACCESS_KEY", "hello")

neu = Neuron()


async def response_stream(data):
    try:
        last_message = data["messages"][-1]
        async for response in neu.scraper_validator.organic(last_message):
            yield f"{response}"

    except Exception as e:
        bt.logging.error(f"error in response_stream {traceback.format_exc()}")


async def response_stream_event(data):
    try:
        last_message = data["messages"][-1]
        uids = None
        if "uids" in data:
            uids = [uid for uid in data["uids"] if uid is not None]
        if uids:
            uids = [uid for uid in data["uids"] if uid is not None]
            print(f"Check uids, {uids}")
            merged_chunks = ""
            async for response in neu.scraper_validator.organic_specified(
                last_message, uids
            ):
                chunk = str(response)  # Assuming response is already a string
                merged_chunks += chunk
                lines = chunk.split("\n")
                sse_data = "\n".join(f"data: {line if line else ' '}" for line in lines)
                yield f"{sse_data}\n\n"
        else:
            uids = None
            merged_chunks = ""
            async for response in neu.scraper_validator.organic(last_message):
                # Decode the chunk if necessary and merge
                chunk = str(response)  # Assuming response is already a string
                merged_chunks += chunk
                lines = chunk.split("\n")
                sse_data = "\n".join(f"data: {line if line else ' '}" for line in lines)
                # print("sse_data: ", sse_data)
                yield f"{sse_data}\n\n"
        # Here you might want to do something with merged_chunks
        # after the loop has finished
    except Exception as e:
        bt.logging.error(f"error in response_stream {traceback.format_exc()}")
        yield f"data: {json.dumps({'error': 'An error occurred'})}\n\n"


@app.post("/analyse-tweets", include_in_schema=False)
async def process_scraper_validator(request: Request, data: dict):
    # Check access key
    access_key = request.headers.get("access-key")
    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")
    return StreamingResponse(response_stream(data))


@app.post("/analyse-tweets-event", include_in_schema=False)
async def process_scraper_validator(request: Request, data: dict):
    # Check access key
    # access_key = request.headers.get("access-key")
    # if access_key != EXPECTED_ACCESS_KEY:
    #     raise HTTPException(status_code=401, detail="Invalid access key")
    return StreamingResponse(response_stream_event(data))


class SearchRequest(BaseModel):
    tools: List[str] = Field(
        ...,
        example=[
            "Google Search",
            "Google Image Search",
            "Hacker News Search",
            "Reddit Search",
        ],
    )
    query: str = Field(..., example="What are the recent sport events?")


available_tools = [
    # SerpGoogleSearchTool(),
    # SerpGoogleImageSearchTool(),
    # HackerNewsSearchTool(),
    # RedditSearchTool(),
]


@app.get(
    "/search",
    response_model=dict,
    summary="Search across multiple platforms",
    description="Performs a search across specified tools",
    response_description="A dictionary of search results from the specified tools.",
    responses={
        200: {
            "description": "A dictionary of search results from the specified tools.",
            "content": {
                "application/json": {
                    "example": {tool.name: {} for tool in available_tools}
                }
            },
        }
    },
)
async def search(
    tools: str = Query(
        ...,
        description="A JSON encoded list of tools to search with",
        example=json.dumps([tool.name for tool in available_tools]),
    ),
    query: str = Query(..., example="What are the recent sport events?"),
):
    tools = json.loads(tools)

    tool_tasks = []

    for tool_name in tools:
        tool_instance = next(
            (tool for tool in available_tools if tool.name == tool_name), None
        )
        if not tool_instance:
            raise HTTPException(status_code=400, detail=f"Invalid tool: {tool_name}")
        tool_tasks.append(asyncio.create_task(tool_instance.ainvoke({"query": query})))

    results = await asyncio.gather(*tool_tasks)

    response = {}
    for tool_name, result in zip(tools, results):
        response[tool_name] = result

    return response


@app.get("/")
async def health_check():
    return {"status": "healthy"}


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8005, timeout_keep_alive=300)


if __name__ == "__main__":
    asyncio.get_event_loop().create_task(neu.run())
    run_fastapi()
