import bittensor as bt
from starlette.types import Send
from datura.protocol import SearchSynapse
from datura.tools.search.serp_google_search_tool import SerpGoogleSearchTool
from datura.tools.search.serp_google_image_search_tool import SerpGoogleImageSearchTool
from datura.tools.hacker_news.hacker_news_search_tool import HackerNewsSearchTool
from datura.tools.reddit.reddit_search_tool import RedditSearchTool
import asyncio


class SearchMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def search(self, synapse: SearchSynapse):
        query = synapse.query
        tools = synapse.tools

        available_tools = [
            SerpGoogleSearchTool(),
            SerpGoogleImageSearchTool(),
            HackerNewsSearchTool(),
            RedditSearchTool(),
        ]

        tool_tasks = []

        for tool_name in tools:
            tool_instance = next(
                (tool for tool in available_tools if tool.name == tool_name), None
            )
            if not tool_instance:
                bt.logging.error(f"Invalid tool: {tool_name}")
            tool_tasks.append(
                asyncio.create_task(tool_instance.ainvoke({"query": query}))
            )

        results = await asyncio.gather(*tool_tasks)

        response = {}

        for tool_name, result in zip(tools, results):
            response[tool_name] = result

        synapse.results = response

        return synapse
