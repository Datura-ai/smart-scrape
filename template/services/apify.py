import os
from apify_client import ApifyClient
from typing import List, Optional
from template.protocol import StreamPrompting, IsAlive, TwitterScraper, TwitterQueryResult
import asyncio

APIFY_API_TOKEN = 'apify_api_tjXL9pd5iVJ84UvYnDAR98JJPTExRx3GcY61' #os.environ.get('APIFY_API_KEY')
APIFY_ACTOR_ID =  '2s3kSMq7tpuC3bI6M' #os.environ.get('APIFY_ACTOR_ID')

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
    run = client.actor(APIFY_ACTOR_ID).call(run_input=actor_input.__dict__)
    
    async for item in client.dataset(run['defaultDatasetId']).iterate_items():
        db.create_or_update_document(item['id'], item)

    return True


def run_actor_based_query_result(query_result: TwitterQueryResult):
    search_queries = [
        # *query_result.hashtags #, *query_result.keywords, *query_result.user_mentions
        "elon mask", "#elon", "spacex"
    ]
    # actor_input: ActorInput = ActorInput(search_queries=search_queries)

    run_input = {
        "exclude_images": False,
        "exclude_links": False,
        "exclude_media": False,
        "exclude_native_retweets": False,
        "exclude_native_video": False,
        "exclude_news": False,
        "exclude_pro_video": False,
        "exclude_quote": False,
        "exclude_replies": False,
        "exclude_safe": False,
        "exclude_verified": False,
        "exclude_videos": False,
        "images": False,
        "include_user_info": True,
        "language": "any",
        "links": False,
        "media": False,
        "native_retweets": False,
        "native_video": False,
        "news": False,
        "pro_video": False,
        "proxy_config": {
            "use_apify_proxy": True,
            "apify_proxy_groups": [
                "RESIDENTIAL"
            ]
        },
        "quote": False,
        "replies": False,
        "safe": False,
        "search_queries": [
            "elon mask",
            "#elon"
        ],
        "tweets_desired": 20,
        "verified": False,
        "videos": False,
        "min_replies": 0,
        "min_retweets": 0,
        "min_likes": 0,
        "from_these_accounts": [],
        "to_these_accounts": [],
        "mentioning_these_accounts": []
    }
    client = ApifyClient(APIFY_API_TOKEN)
    
    # Start the actor and wait for it to finish
    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
    
    items = []
   # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        print(item)

    print(len(items), "Parsed Items lentght")
    return items



# Example usage:
import asyncio

search_queries = ["elon mask", "#elon", "spacex"]
actor_input = ActorInput(search_queries=search_queries)

async def main():
    items = run_actor_based_query_result(None)
    for item in items:
        print(item)

asyncio.run(main())




