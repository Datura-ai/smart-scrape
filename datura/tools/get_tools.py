from typing import List
from datura.tools.base import BaseToolkit, BaseTool
from datura.tools.twitter.twitter_toolkit import TwitterToolkit
from datura.tools.search.search_toolkit import SearchToolkit
from datura.tools.discord.discord_toolkit import DiscordToolkit
from datura.tools.reddit.reddit_toolkit import RedditToolkit
from datura.tools.hacker_news.hacker_news_toolkit import HackerNewsToolkit
from datura.tools.bittensor.bittensor_toolkit import BittensorToolkit
from datura.tools.subnets_source_code.subnets_source_code_toolkit import SubnetsSourceCodeToolkit

TOOLKITS: List[BaseToolkit] = [
    SearchToolkit(),
    TwitterToolkit(),
    DiscordToolkit(),
    RedditToolkit(),
    HackerNewsToolkit(),
    BittensorToolkit(),
    SubnetsSourceCodeToolkit()
]


def get_all_tools():
    """Return a list of all tools."""
    result: List[BaseTool] = []

    for toolkit in TOOLKITS:
        tools = toolkit.get_tools()
        result.extend(tools)

    return result


def find_toolkit_by_tool_name(tool_name: str):
    """Return the toolkit that contains the tool with the given name."""
    for toolkit in TOOLKITS:
        for tool in toolkit.get_tools():
            if tool.name == tool_name:
                return toolkit

    return None


def find_toolkit_by_name(toolkit_name: str):
    """Return the toolkit with the given name."""
    for toolkit in TOOLKITS:
        if toolkit.name == toolkit_name:
            return toolkit

    return None
