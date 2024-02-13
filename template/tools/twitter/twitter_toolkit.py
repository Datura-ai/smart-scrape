from abc import ABC
from typing import List
from template.tools.base import BaseToolkit, BaseTool, ToolEnvKey
from template.tools.twitter.get_recent_tweets_tool import GetRecentTweetsTool
from template.tools.twitter.get_full_archive_tweets_tool import GetFullArchiveTweetsTool


class TwitterToolkit(BaseToolkit, ABC):
    name: str = "Twitter Toolkit"
    description: str = "Toolkit containing tools for tweeting"
    slug: str = "twitter"
    toolkit_id = "0e0ae6fb-0f1c-4d00-bc84-1feb2a6824c6"

    is_active = False

    def get_tools(self) -> List[BaseTool]:
        return [GetRecentTweetsTool(), GetFullArchiveTweetsTool()]

    def get_env_keys(self) -> List[ToolEnvKey]:
        return []
