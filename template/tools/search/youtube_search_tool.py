from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from youtube_search import YoutubeSearch

from template.tools.base import BaseTool


class YoutubeSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Youtube search.",
    )


class YoutubeSearchTool(BaseTool):
    """Tool for the Youtube API."""

    name = "Youtube Search"

    slug = "youtube-search"

    description = (
        "Useful for when you need to search videos on Youtube"
        "Input should be a search query."
    )

    args_schema: Type[YoutubeSearchSchema] = YoutubeSearchSchema

    tool_id = "8b7b6dad-e550-4a01-be51-aed785eda805"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search Youtube and return the results."""
        result = YoutubeSearch(search_terms=query, max_results=10)
        return result.videos

    async def send_event(self, send, response_streamer, data):
        pass
