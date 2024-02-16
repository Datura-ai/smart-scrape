import os
from typing import List
import bittensor as bt
from apify_client import ApifyClientAsync
from template.protocol import TwitterScraperTweet

APIFY_API_KEY = os.environ.get("APIFY_API_KEY")

#todo at ths moment just warning, later it will be required
# if not APIFY_API_KEY:
#     raise ValueError(
#         "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md"
#     )


class TwitterScraperActor:
    def __init__(self) -> None:
        # Actor: https://apify.com/quacker/twitter-url-scraper
        self.actor_id = "u6ppkMWAx2E2MpEuF"
        self.client = ApifyClientAsync(token=APIFY_API_KEY)

    async def get_tweets(
        self, urls: List[str], add_user_info: bool = True
    ) -> List[TwitterScraperTweet]:
        if not APIFY_API_KEY:
            bt.logging.warning("Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release.")
            return []
        try:
            run_input = {
                "startUrls": [{"url": url} for url in urls],
                "proxyConfig": {"useApifyProxy": True},
                "addUserInfo": add_user_info,
            }

            run = await self.client.actor(self.actor_id).call(run_input=run_input)

            tweets: List[TwitterScraperTweet] = []

            async for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                tweet = TwitterScraperTweet(**item)
                tweets.append(tweet)

            return tweets
        except Exception as e:
            bt.logging.warning(f"Failed to scrape tweets: {e}")
            return []
