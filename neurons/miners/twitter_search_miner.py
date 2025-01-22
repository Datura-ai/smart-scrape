import bittensor as bt
from datura.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
    Model,
)


class TwitterSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def search(self, synapse: TwitterSearchSynapse):
        # Extract the query parameters from the synapse
        query = synapse.query
        search_params = {
            "sort": synapse.sort,
            "start_date": synapse.start_date,
            "end_date": synapse.end_date,
            "lang": synapse.lang,
            "verified": synapse.verified,
            "blue_verified": synapse.blue_verified,
            "is_quote": synapse.is_quote,
            "is_video": synapse.is_video,
            "is_image": synapse.is_image,
            "min_retweets": synapse.min_retweets,
            "min_replies": synapse.min_replies,
            "min_likes": synapse.min_likes,
        }

        # Log query and search parameters
        bt.logging.info(
            f"Executing mock search with query: {query} and params: {search_params}"
        )

        # Mock tweet result
        mock_tweet = {
            "user": {"username": "mock_user", "verified": True},
            "id": "123456789",
            "text": "This is a mock tweet for testing purposes.",
            "reply_count": 10,
            "retweet_count": 5,
            "like_count": 50,
            "view_count": 100,
            "quote_count": 1,
            "impression_count": 200,
            "bookmark_count": 2,
            "url": "https://twitter.com/mock_user/status/123456789",
            "created_at": "2025-01-13T12:00:00Z",
            "media": [],
            "is_quote_tweet": False,
            "is_retweet": False,
        }

        # Assign the mock tweet to the results field of the synapse
        synapse.results = [mock_tweet]

        bt.logging.info(f"here is the final synapse {synapse}")
        return synapse

    async def search_by_id(self, synapse: TwitterIDSearchSynapse):
        """
        Perform a Twitter search based on a specific tweet ID.
        """
        tweet_id = synapse.id

        # Log the search operation
        bt.logging.info(f"Searching for tweet by ID: {tweet_id}")

        # Mock result for the given tweet ID
        mock_tweet = {
            "user": {"username": "mock_user", "verified": True},
            "id": tweet_id,
            "text": f"This is a mock tweet for ID: {tweet_id}",
            "reply_count": 5,
            "retweet_count": 15,
            "like_count": 30,
            "view_count": 200,
            "quote_count": 0,
            "impression_count": 300,
            "bookmark_count": 1,
            "url": f"https://twitter.com/mock_user/status/{tweet_id}",
            "created_at": "2025-01-13T12:00:00Z",
            "media": [],
            "is_quote_tweet": False,
            "is_retweet": False,
        }

        # Assign the mock tweet to the results field of the synapse
        synapse.results = [mock_tweet]

        return synapse

    async def search_by_urls(self, synapse: TwitterURLsSearchSynapse):
        """
        Perform a Twitter search based on multiple tweet URLs.

        Parameters:
            synapse (TwitterURLsSearchSynapse): Contains the list of tweet URLs.

        Returns:
            TwitterURLsSearchSynapse: The synapse with fetched tweets in the results field.
        """
        urls = synapse.urls

        # Log the search operation
        bt.logging.info(f"Searching for tweets by URLs: {urls}")

        # Mock results for the given URLs
        mock_results = []
        for url in urls:

            mock_tweet = {
                "user": {"username": "mock_user", "verified": True},
                "id": "12",
                "text": f"This is a mock tweet for the URL: {url}",
                "reply_count": 10,
                "retweet_count": 20,
                "like_count": 40,
                "view_count": 300,
                "quote_count": 2,
                "impression_count": 400,
                "bookmark_count": 3,
                "url": url,
                "created_at": "2025-01-13T12:00:00Z",
                "media": [],
                "is_quote_tweet": False,
                "is_retweet": False,
            }
            mock_results.append(mock_tweet)

        # Assign the mock tweets to the results field of the synapse
        synapse.results = mock_results

        return synapse
