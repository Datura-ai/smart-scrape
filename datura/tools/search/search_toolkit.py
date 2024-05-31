from abc import ABC
from typing import List
from datura.tools.base import BaseToolkit, BaseTool
from .serp_google_search_tool import SerpGoogleSearchTool
from .serp_google_image_search_tool import SerpGoogleImageSearchTool
from .serp_bing_search_tool import SerpBingSearchTool
from .wikipedia_search_tool import WikipediaSearchTool
from .youtube_search_tool import YoutubeSearchTool
from .arxiv_search_tool import ArxivSearchTool
from .serp_google_news_search_tool import SerpGoogleNewsSearchTool
from .search_summary import summarize_search_data, prepare_search_data_for_summary


TOOLS = [
    SerpGoogleSearchTool(),
    SerpGoogleImageSearchTool(),
    SerpBingSearchTool(),
    WikipediaSearchTool(),
    YoutubeSearchTool(),
    ArxivSearchTool(),
    SerpGoogleNewsSearchTool(),
]


class SearchToolkit(BaseToolkit, ABC):
    name: str = "Search Toolkit"
    description: str = (
        "Toolkit containing tools for performing web, youtube, wikipedia and other searches."
    )

    slug: str = "web-search"
    toolkit_id = "fed46dde-ee8e-420b-a1bb-4a161aa01dca"

    def get_tools(self) -> List[BaseTool]:
        return TOOLS

    async def summarize(self, prompt, model, data):
        response_order = self.tool_manager.response_order
        return await summarize_search_data(
            prompt=prompt,
            model=model,
            data=prepare_search_data_for_summary(data),
            response_order=response_order,
        )
