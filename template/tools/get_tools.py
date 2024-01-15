from typing import List

from template.tools.base import BaseTool, BaseToolkit
from tools.serp.serp_google_search_toolkit import SerpGoogleSearchToolkit
from tools.twitter.twitter_toolkit import TwitterToolkit
from serp import SerpGoogleSearchToolkit

TOOLKITS: List[BaseToolkit] = [
    SerpGoogleSearchToolkit(),
    TwitterToolkit()
]

def get_all_tools():
    """Return a list of all tools."""
    result = []

    for toolkit in TOOLKITS:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.slug,
                    "description": tool.description,
                    "parameters": tool.get_params(),
                },
            }
            for tool in toolkit.get_tools()
        ]
        result.extend(tools)

    return result

def get_avalaible_functions():
    """Return a list of all tools."""
    result = []

    for toolkit in TOOLKITS:
        tools = [
            {
                [tool.slug]: tool._run,
            }
            for tool in toolkit.get_tools()
        ]
        result.extend(tools)

    return result
