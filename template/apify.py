import os
from apify_client import ApifyClient
from typing import List, Optional
from .db import DBClient


APIFY_API_TOKEN = os.environ.get('APIFY_API_KEY')
APIFY_ACTOR_ID =  os.environ.get('APIFY_ACTOR_ID')

class ProxyConfig:
    def __init__(self, use_apify_proxy: bool = True, apify_proxy_groups: List[str] = ["RESIDENTIAL"]):
        self.use_apify_proxy = use_apify_proxy
        self.apify_proxy_groups = apify_proxy_groups

class ActorInput:
    def __init__(
        self,
        search_queries: List[str],
        tweets_desired: int = 100,
        include_user_info: bool = True,
        min_replies: int = 0,
        min_retweets: int = 0,
        min_likes: int = 0,
        from_these_accounts: Optional[List[str]] = None,
        to_these_accounts: Optional[List[str]] = None,
        mentioning_these_accounts: Optional[List[str]] = None,
        native_retweets: bool = False,
        media: bool = False,
        images: bool = False,
        videos: bool = False,
        news: bool = False,
        verified: bool = False,
        native_video: bool = False,
        replies: bool = False,
        links: bool = False,
        safe: bool = False,
        quote: bool = False,
        pro_video: bool = False,
        exclude_native_retweets: bool = False,
        exclude_media: bool = False,
        exclude_images: bool = False,
        exclude_videos: bool = False,
        exclude_news: bool = False,
        exclude_verified: bool = False,
        exclude_native_video: bool = False,
        exclude_replies: bool = False,
        exclude_links: bool = False,
        exclude_safe: bool = False,
        exclude_quote: bool = False,
        exclude_pro_video: bool = False,
        language: str = "any",
        proxy_config: ProxyConfig = ProxyConfig()
    ):
        self.search_queries = search_queries
        self.tweets_desired = tweets_desired
        self.include_user_info = include_user_info
        self.min_replies = min_replies
        self.min_retweets = min_retweets
        self.min_likes = min_likes
        self.from_these_accounts = from_these_accounts or []
        self.to_these_accounts = to_these_accounts or []
        self.mentioning_these_accounts = mentioning_these_accounts or []
        self.native_retweets = native_retweets
        self.media = media
        self.images = images
        self.videos = videos
        self.news = news
        self.verified = verified
        self.native_video = native_video
        self.replies = replies
        self.links = links
        self.safe = safe
        self.quote = quote
        self.pro_video = pro_video
        self.exclude_native_retweets = exclude_native_retweets
        self.exclude_media = exclude_media
        self.exclude_images = exclude_images
        self.exclude_videos = exclude_videos
        self.exclude_news = exclude_news
        self.exclude_verified = exclude_verified
        self.exclude_native_video = exclude_native_video
        self.exclude_replies = exclude_replies
        self.exclude_links = exclude_links
        self.exclude_safe = exclude_safe
        self.exclude_quote = exclude_quote
        self.exclude_pro_video = exclude_pro_video
        self.language = language
        self.proxy_config = proxy_config.__dict__


async def run_actor(actor_input: ActorInput):
    client = ApifyClient(APIFY_API_TOKEN)
    run = await client.actor(APIFY_ACTOR_ID).call(run_input=actor_input.__dict__)
    return run

async def run_actor_and_store_data(actor_input: ActorInput):
    client = ApifyClient(APIFY_API_TOKEN)
    # Start the actor and wait for it to finish
    run = await client.actor(APIFY_ACTOR_ID).call(run_input=actor_input.__dict__)
    
    # Fetch and print Actor results from the run's dataset (if there are any)
    db = DBClient()
    async for item in client.dataset(run['defaultDatasetId']).iterate_items():
        db.create_or_update_document(item['id'], item)

    return True



# # Example usage:
# import asyncio

# search_queries = ["#openai"]
# actor_input = ActorInput(search_queries=search_queries)

# async def main():
#     items = await run_actor_and_fetch_results(actor_id, actor_input)
#     for item in items:
#         print(item)

# asyncio.run(main())


