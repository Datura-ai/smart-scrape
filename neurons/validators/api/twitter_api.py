from fastapi import APIRouter, HTTPException
import aiohttp
import os
import bittensor as bt

BASE_URL = "https://api.twitter.com/2/users"
BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

twitter_api_router = APIRouter(
    prefix='/twitter',
)

async def bearer_oauth(session: aiohttp.ClientSession):
    session.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    session.headers["User-Agent"] = "v2RecentSearchPython"

async def connect_to_endpoint(url, params):
    async with aiohttp.ClientSession() as session:
        await bearer_oauth(session)
        async with session.get(url, params=params) as response:
            if response.status in [401, 403]:
                bt.logging.error(
                    f"Critical Twitter API Request error occurred: {await response.text()}"
                )

            json_data = None

            try:
                json_data = await response.json()
            except aiohttp.ContentTypeError:
                pass

            response_text = await response.text()
            return json_data, response.status, response_text

@twitter_api_router.get('/users/{user_id}/following')
async def fetch_followings(user_id: str):
    url = f'{BASE_URL}/{user_id}/following'
    response_json, status_code, response_text = await connect_to_endpoint(url, {})
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=response_text)
    return response_json


@twitter_api_router.get('/users/{user_id}')
async def get_user_by_id(user_id: str, user_fields: str):
    url = f'{BASE_URL}/{user_id}'
    params = {
        "user.fields": user_fields
    }
    response_json, status_code, response_text = await connect_to_endpoint(url, params)
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=response_text)
    return response_json

@twitter_api_router.get('/users/by/username/{username}')
async def get_user_by_username(username: str, user_fields: str):
    # https://api.twitter.com/2/users/by/username/$%7Busername%7D?
    url = f'{BASE_URL}/by/username/{username}'
    params = {
        "user.fields": user_fields
    }
    
    response_json, status_code, response_text = await connect_to_endpoint(url, params)
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=response_text)
    
    return response_json

