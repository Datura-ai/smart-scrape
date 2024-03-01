from abc import ABC
from typing import List
from template.tools.base import BaseToolkit, BaseTool
from .serp_google_search_tool import SerpGoogleSearchTool
from .wikipedia_search_tool import WikipediaSearchTool
from .youtube_search_tool import YoutubeSearchTool
from .arxiv_search_tool import ArxivSearchTool
from .search_summary import summarize_search_data


class SearchToolkit(BaseToolkit, ABC):
    name: str = "Search Toolkit"
    description: str = (
        "Toolkit containing tools for performing web, youtube, wikipedia and other searches."
    )

    slug: str = "web-search"
    toolkit_id = "fed46dde-ee8e-420b-a1bb-4a161aa01dca"

    def get_tools(self) -> List[BaseTool]:
        return [
            SerpGoogleSearchTool(),
            WikipediaSearchTool(),
            YoutubeSearchTool(),
            ArxivSearchTool(),
        ]

    async def summarize(self, prompt, model, data):
        return await summarize_search_data(prompt=prompt, model=model, data=data)
