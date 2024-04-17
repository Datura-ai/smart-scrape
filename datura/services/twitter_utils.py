import re
from typing import List
from urllib.parse import urlparse

VALID_DOMAINS = ["twitter.com", "x.com"]


class TwitterUtils:
    def __init__(self):
        self.twitter_link_regex = re.compile(
            r"https?://(?:"
            + "|".join(re.escape(domain) for domain in VALID_DOMAINS)
            + r")/(?![^/]*?(?:Twitter|Admin)[^/]*?/)"
            r"(?P<username>[a-zA-Z0-9_]{1,15})/status/(?P<id>\d+)",
            re.IGNORECASE,
        )

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

    @staticmethod
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
        return [match.group() for match in self.twitter_link_regex.finditer(text)]
