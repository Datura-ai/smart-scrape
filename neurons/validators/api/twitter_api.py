from fastapi import APIRouter, HTTPException
from datura.services.twitter_api_wrapper import TwitterAPIClient

twitter_api_router = APIRouter(
    prefix='/twitter',
)

client = TwitterAPIClient()


@twitter_api_router.get('/users/{user_id}/following')
async def get_user_followings(user_id: str):
    response_json, status_code, response_text = await client.get_user_followings(
        user_id
    )
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=response_text)
    return response_json


@twitter_api_router.get('/users/{user_id}')
async def get_user_by_id(user_id: str, user_fields: str):
    params = {
        "user.fields": user_fields
    }
    response_json, status_code, response_text = await client.get_user(
        user_id,
        params,
    )
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=response_text)
    return response_json


@twitter_api_router.get('/users/by/username/{username}')
async def get_user_by_username(username: str, user_fields: str):
    params = {
        "user.fields": user_fields
    }

    response_json, status_code, response_text = await client.get_user_by_username(
        username, params
    )
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=response_text)
    return response_json
