import requests
import os
import re
import bittensor as bt
from typing import List
from urllib.parse import urlparse

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

VALID_DOMAINS = ["twitter.com", "x.com"]


class TwitterAPIClient:
    def __init__(
        self,
        openai_query_model="gpt-3.5-turbo-1106",
        openai_fix_query_model="gpt-4-1106-preview",
    ):
        # self.bearer_token = os.environ.get("BEARER_TOKEN")
        self.bearer_token = BEARER_TOKEN
        self.twitter_link_regex = re.compile(
            r"https?://(?:"
            + "|".join(re.escape(domain) for domain in VALID_DOMAINS)
            + r")/[\w/:%#\$&\?\(\)~\.=\+\-]+(?<![\.\)])",
            re.IGNORECASE,
        )
        self.openai_query_model = openai_query_model
        self.openai_fix_query_model = openai_fix_query_model

    def bearer_oauth(self, r):
        """
        Method required by bearer token authentication.
        """
        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2RecentSearchPython"
        return r

    def connect_to_endpoint(self, url, params):
        response = requests.get(url, auth=self.bearer_oauth, params=params)

        if response.status_code in [401, 403]:
            bt.logging.error(
                f"Critical Twitter API Ruquest error occurred: {response.text}"
            )
            os._exit(1)

        return response

    def get_tweet_by_id(self, tweet_id):
        tweet_url = f"https://api.twitter.com/2/tweets/{tweet_id}"
        response = self.connect_to_endpoint(tweet_url, {})
        if response.status_code != 200:
            return None
        return response.json()

    def get_tweets_by_ids(self, tweet_ids):
        ids = ",".join(tweet_ids)  # Combine all tweet IDs into a comma-separated string
        tweets_url = f"https://api.twitter.com/2/tweets?ids={ids}"
        response = self.connect_to_endpoint(tweets_url, {})
        if response.status_code != 200:
            return []
        return response.json()

    def get_recent_tweets(self, query_params):
        search_url = "https://api.twitter.com/2/tweets/search/recent"
        response = self.connect_to_endpoint(search_url, query_params)
        return response

    def get_full_archive_tweets(self, query_params):
        search_url = "https://api.twitter.com/2/tweets/search/all"
        response = self.connect_to_endpoint(search_url, query_params)
        return response

    @staticmethod
    def extract_tweet_id(url: str) -> str:
        """
        Extract the tweet ID from a Twitter URL.

        Args:
            url: The Twitter URL to extract the tweet ID from.

        Returns:
            The extracted tweet ID.
        """
        match = re.search(r"/status(?:es)?/(\d+)", url)
        return match.group(1) if match else None

    def fetch_twitter_data_for_links(self, links: List[str]) -> List[dict]:
        tweet_ids = [
            self.extract_tweet_id(link)
            for link in links
            if self.is_valid_twitter_link(link)
        ]
        return self.get_tweets_by_ids(tweet_ids)

    def is_valid_twitter_link(self, url: str) -> bool:
        """
        Check if the given URL is a valid Twitter link.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a valid Twitter link, False otherwise.
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc.lower() in VALID_DOMAINS

    def find_twitter_links(self, text: str) -> List[str]:
        """
        Find all Twitter links in the given text.

        Args:
            text: The text to search for Twitter links.

        Returns:
            A list of found Twitter links.
        """
        return self.twitter_link_regex.findall(text)
