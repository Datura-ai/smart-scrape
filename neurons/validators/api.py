import os
from typing import Optional
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request, Query, APIRouter
from datura.protocol import TwitterAPISynapseCall
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

        max_execution_time = int(data.get("max_execution_time", 10))

        if max_execution_time not in [10, 30, 120]:
            raise ValueError("max_execution_time must be either 10 or 120.")

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
            async for response in neu.scraper_validator.organic(
                last_message, max_execution_time
            ):
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

SEARCH_DESCRIPTION = """Performs a search across multiple platforms. Available tools are:
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
"""


@app.get(
    "/search",
    summary="Search across multiple platforms",
    description=SEARCH_DESCRIPTION,
    response_description="A dictionary of search results from the specified tools.",
    responses={
        200: {
            "description": "A dictionary of search results from the specified tools.",
            "content": {
                "application/json": {"example": {tool: {} for tool in available_tools}}
            },
        }
    },
)
async def search(
    tools: str = Query(
        ...,
        description="A JSON encoded list of tools to search with",
        example=json.dumps([tool for tool in available_tools]),
    ),
    query: str = Query(..., example="What are the recent sport events?"),
    uid: Optional[int] = Query(
        None,
        example=0,
        description="Optional miner uid to run. If not provided, a random miner will be selected.",
    ),
):
    tools = json.loads(tools)

    try:
        result = await neu.scraper_validator.search(query, tools, uid)
        return result
    except Exception as e:
        bt.logging.error(f"error in search {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred, {e}")


twitter_api_router = APIRouter(prefix="/twitter")


@twitter_api_router.get(
    "/users/{user_id}/following",
    summary="Get User Followings",
    description="Retrieve the list of users that the specified user is following on Twitter.",
    response_description="A list of users",
    responses={
        200: {
            "description": "A list of users",
            "content": {"application/json": {"example": {"data": []}}},
        }
    },
)
async def get_user_followings(
    user_id: str,
    max_items: Optional[int] = Query(
        20,
        alias="max_items",
        description="Maximum number of users to return per query",
    ),
):
    try:
        response = await neu.scraper_validator.get_twitter_user(
            body={
                "user_id": user_id,
                "request_type": TwitterAPISynapseCall.GET_USER_FOLLOWINGS,
                "max_items": max_items,
            }
        )
        return response
    except Exception as e:
        bt.logging.error(
            f"Error in get_user_followings for for user: {user_id}: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"An error occurred, {e}")


@twitter_api_router.get(
    "/users/{user_id}/followers",
    summary="Get User Followers",
    description="Retrieve specified user's followers as list",
    response_description="A list of users",
    responses={
        200: {
            "description": "A list of users",
            "content": {"application/json": {"example": {"data": []}}},
        }
    },
)
async def get_user_followers(
    user_id: str,
    max_items: Optional[int] = Query(
        20,
        alias="max_items",
        description="Maximum number of users to return per query",
    ),
):
    try:
        response = await neu.scraper_validator.get_twitter_user(
            body={
                "user_id": user_id,
                "request_type": TwitterAPISynapseCall.GET_USER_FOLLOWERS,
                "max_items": max_items,
            }
        )
        return response
    except Exception as e:
        bt.logging.error(
            f"Error in get_user_followers for user: {user_id}: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"An error occurred, {e}")


@twitter_api_router.get(
    "/users/{user_id}",
    summary="Get User by ID",
    description="Retrieve the details of a Twitter user by their user ID.",
    response_description="The details of the specified Twitter user.",
    responses={
        200: {
            "description": "The details of the specified Twitter user.",
            "content": {"application/json": {"example": {"data": {}}}},
        }
    },
)
async def get_user_by_id(
    user_id: str,
):
    try:
        response = await neu.scraper_validator.get_twitter_user(
            body={
                "user_id": user_id,
                "request_type": TwitterAPISynapseCall.GET_USER,
            }
        )
        return response
    except Exception as e:
        bt.logging.error(
            f"Error in get_user_followings for user: {user_id}: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"An error occurred, {e}")


@twitter_api_router.get(
    "/users/by/username/{username}",
    summary="Get User by Username",
    description="Retrieve the details of a Twitter user by their username.",
    response_description="The details of the specified Twitter user.",
    responses={
        200: {
            "description": "The details of the specified Twitter user.",
            "content": {"application/json": {"example": {"data": {}}}},
        }
    },
)
async def get_user_by_username(
    username: str,
):
    try:
        response = await neu.scraper_validator.get_twitter_user(
            body={
                "username": username,
                "request_type": TwitterAPISynapseCall.GET_USER_WITH_USERNAME,
            }
        )
        return response
    except Exception as e:
        bt.logging.error(
            f"Error in get_user_followings for user: {username}: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"An error occurred, {e}")


@twitter_api_router.get(
    "/search",
    summary="Search Tweets",
    description="Search",
    response_description="List of tweets returned from search query",
    responses={
        200: {
            "description": "List of tweets returned from search query",
            "content": {"application/json": {"example": {"data": {}}}},
        }
    },
)
async def search_twitter(
    prompt: str = Query(
        "",
        alias="query",
        description="Query text to search tweets with",
    ),
    max_items: Optional[int] = Query(
        20, alias="max_items", description="Max items to be returned per query"
    ),
    min_retweets: Optional[int] = Query(
        None,
        alias="min_retweets",
        description="Filter to get tweets with minimum number of retweets",
    ),
    min_likes: Optional[int] = Query(
        None,
        alias="min_likes",
        description="Filter to get tweets with minimum number of likes",
    ),
    only_verified: Optional[bool] = Query(
        False,
        alias="only_verified",
        description="Filter to get only verified users' tweets",
    ),
    only_twitter_blue: Optional[bool] = Query(
        False,
        alias="only_twitter_blue",
        description="Filter to get only twitter blue users' tweets",
    ),
    only_video: Optional[bool] = Query(
        False,
        alias="only_video",
        description="Filter to get only those tweets which has video embedded",
    ),
    only_image: Optional[bool] = Query(
        False,
        alias="only_image",
        description="Filter to get only those tweets which has image embedded",
    ),
    only_quote: Optional[bool] = Query(
        False,
        alias="only_quote",
        description="Filter to get only those tweets which has quote embedded",
    ),
    start_date: Optional[str] = Query(
        None,
        alias="start_date",
        description="Date range field for tweet, combine with end_date field to set a time range",
    ),
    end_date: Optional[str] = Query(
        None,
        alias="end_date",
        description="Date range field for tweet, combine with start_date field to set a time range",
    ),
):
    try:
        response = await neu.scraper_validator.get_tweets(
            body={
                "prompt": prompt,
                "max_items": max_items,
                "min_retweets": min_retweets,
                "min_likes": min_likes,
                "only_verified": only_verified,
                "only_twitter_blue": only_twitter_blue,
                "only_video": only_video,
                "only_image": only_image,
                "only_quote": only_quote,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return response
    except Exception as e:
        bt.logging.error(
            f"Error in search_twitter for prompt: {prompt}: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"An error occurred, {e}")


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
        servers=[{"url": "http://localhost:8005", "description": "Datura API"}],
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.include_router(twitter_api_router)

app.openapi = custom_openapi


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8005, timeout_keep_alive=300)


if __name__ == "__main__":
    asyncio.get_event_loop().create_task(neu.run())
    run_fastapi()
