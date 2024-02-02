import os
from typing import List
from apify_client import ApifyClient
from template.protocol import TwitterScraperTweet

APIFY_API_KEY = os.environ.get("APIFY_API_KEY")

if not APIFY_API_KEY:
    raise ValueError(
        "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md"
    )


class TwitterScraperActor:
    def __init__(self) -> None:
        self.actor_id = "KVJr35xjTw2XyvMeK"
        self.client = ApifyClient(token=APIFY_API_KEY)

    def get_tweets(self, urls: List[str], add_user_info: bool = True):
        run_input = {
            "startUrls": [{"url": url} for url in urls],
            "proxyConfig": {"useApifyProxy": True},
            "addUserInfo": add_user_info,
        }

        run = self.client.actor(self.actor_id).call(run_input=run_input)

        tweets = []

        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            tweet = TwitterScraperTweet(**item)
            tweets.append(tweet)

        return tweets
