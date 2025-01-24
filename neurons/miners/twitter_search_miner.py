import bittensor as bt
from datura.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
    Model,
    TwitterScraperTweet,
)
from pydantic import ValidationError


class TwitterSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner

    def _generate_mock_tweet(self, **kwargs):
        """
        Generate a mock tweet using the TwitterScraperTweet class to ensure proper formatting.

        Parameters:
            **kwargs: Fields to override in the default mock tweet.

        Returns:
            dict: Formatted mock tweet as a dictionary.
        """
        try:
            mock_tweet = TwitterScraperTweet(
                user={"username": "mock_user", "verified": True},
                id="123456789",
                text="This is a mock tweet for testing purposes.",
                reply_count=10,
                retweet_count=5,
                like_count=50,
                quote_count=1,
                bookmark_count=2,
                url="https://twitter.com/mock_user/status/123456789",
                created_at="2025-01-13T12:00:00Z",
                media=[],
                is_quote_tweet=False,
                is_retweet=False,
                **kwargs,  # Override any fields with provided values
            )
            return mock_tweet.dict()
        except ValidationError as e:
            bt.logging.error(f"Validation error while creating mock tweet: {e}")
            raise

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

        # Generate a mock tweet
        mock_tweet = self._generate_mock_tweet()

        # Assign the mock tweet to the results field of the synapse
        synapse.results = [mock_tweet]

        bt.logging.info(f"Here is the final synapse: {synapse}")
        return synapse

    async def search_by_id(self, synapse: TwitterIDSearchSynapse):
        """
        Perform a Twitter search based on a specific tweet ID.
        """
        tweet_id = synapse.id

        # Log the search operation
        bt.logging.info(f"Searching for tweet by ID: {tweet_id}")

        # Generate a mock tweet
        mock_tweet = self._generate_mock_tweet(
            id=tweet_id, text=f"This is a mock tweet for ID: {tweet_id}"
        )

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

        # Generate mock tweets for each URL
        mock_results = [
            self._generate_mock_tweet(
                url=url, text=f"This is a mock tweet for the URL: {url}"
            )
            for url in urls
        ]

        # Assign the mock tweets to the results field of the synapse
        synapse.results = mock_results

        return synapse
