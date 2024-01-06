import re
import torch
import json
import time
from typing import List
from urllib.parse import urlparse
from neurons.validators.utils.tasks import TwitterTask
from neurons.validators.penalty import BasePenaltyModel, PenaltyModelType
from template.protocol import TwitterScraperStreaming
from template.services.twitter import TwitterAPIClient 


class LinkValidationPenaltyModel(BasePenaltyModel):
    """
    A class to validate the presence and relevance of Twitter links in a text.
    Inherits from BasePenaltyModel.
    """
    VALID_DOMAINS = ["twitter.com", "x.com"]

    def __init__(self, max_penalty: float):
        """
        Initialize the TwitterLinkValidator with a maximum penalty and the user's prompt content.

        Args:
            max_penalty: The maximum penalty that can be applied to a completion.
            prompt_content: The content of the user's prompt to check relevance.
        """
        super().__init__(max_penalty)
        self.twitter_link_regex = re.compile(r'https?://(?:' + '|'.join(re.escape(domain) for domain in self.VALID_DOMAINS) + r')/[\w/:%#\$&\?\(\)~\.=\+\-]+', re.IGNORECASE)
        self.client = TwitterAPIClient()

    @property
    def name(self) -> str:
        """
        Returns the name of the penalty model.

        Returns:
            The name of the penalty model as defined in PenaltyModelType.
        """
        return PenaltyModelType.link_validation_penalty.value

    def find_twitter_links(self, text: str) -> List[str]:
        """
        Find all Twitter links in the given text.

        Args:
            text: The text to search for Twitter links.

        Returns:
            A list of found Twitter links.
        """
        return self.twitter_link_regex.findall(text)

    def is_valid_twitter_link(self, url: str) -> bool:
        """
        Check if the given URL is a valid Twitter link.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a valid Twitter link, False otherwise.
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc.lower() in self.VALID_DOMAINS
    
    def fetch_twitter_data_for_links(self, links: List[str]) -> List[dict]:
        """
        Retrieve Twitter data for the given list of Twitter links.

        Args:
            links: A list of Twitter links to retrieve data for.

        Returns:
            A list of dictionaries containing the retrieved Twitter data.
        """
        tweet_ids = [self.extract_tweet_id(link) for link in links if self.is_valid_twitter_link(link)]
        return self.client.get_tweets_by_ids(tweet_ids)

    @staticmethod
    def extract_tweet_id(url: str) -> str:
        """
        Extract the tweet ID from a Twitter URL.

        Args:
            url: The Twitter URL to extract the tweet ID from.

        Returns:
            The extracted tweet ID.
        """
        match = re.search(r'/status/(\d+)', url)
        return match.group(1) if match else None


    def calculate_penalties(self, task: TwitterTask, responses: List[TwitterScraperStreaming]) -> torch.FloatTensor:
        """
        Calculates the penalties for each completion based on the presence and relevance of Twitter links.

        Args:
            task: The task containing the query criteria.
            responses: A list of strings representing the completed texts.

        Returns:
            A tensor of penalties for each completion.
        """
        self.prompt_content = task.base_text
        penalties = []
        for response in responses:
            time.sleep(10)
            completion = response.completion
            twitter_links = self.find_twitter_links(completion)
            if twitter_links and all(self.is_valid_twitter_link(link) for link in twitter_links):
                tweets = []
                errors = []
                json_response = self.fetch_twitter_data_for_links(twitter_links)
                if 'data' in json_response:
                    tweets =  json_response['data']
                elif 'errors' in json_response:
                    errors = json_response['errors']
                
                response.tweets = json.dumps(tweets, indent=4, sort_keys=True)
                penalty = self.max_penalty * len(tweets) / len(twitter_links)
                penalties.append(penalty)
            else:
                penalties.append(0.0)
        return torch.tensor(penalties, dtype=torch.float32)
