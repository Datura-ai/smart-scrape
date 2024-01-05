import re
import torch
from typing import List
from urllib.parse import urlparse
from utils.tasks import Task
from penalty import BasePenaltyModel, PenaltyModelType
from template.protocol import TwitterScraperStreaming


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


    def calculate_penalties(self, task: Task, responses: List[TwitterScraperStreaming]) -> torch.FloatTensor:
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
            completion = response.completion
            twitter_links = self.find_twitter_links(completion)
            if not twitter_links or not all(self.is_valid_twitter_link(link) for link in twitter_links):
                penalties.append(self.max_penalty)
            else:
                penalties.append(0.0)
        return torch.tensor(penalties, dtype=torch.float32)
