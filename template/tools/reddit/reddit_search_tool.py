from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import json
import bittensor as bt

from template.services.reddit_api_wrapper import RedditAPIWrapper
from template.tools.search.serp_advanced_google_search import SerpAdvancedGoogleSearch
from template.tools.base import BaseTool


class RedditSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Reddit search.",
    )


class RedditSearchTool(BaseTool):
    """Tool for the Reddit API."""

    name = "Reddit Search"

    slug = "reddit-search"

    description = "A wrapper around Reddit." "Useful for searching Reddit for posts."

    args_schema: Type[RedditSearchSchema] = RedditSearchSchema

    tool_id = "043489f8-ef05-4151-8849-7f954e4910be"

    def _run():
        pass

    async def _arun(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search Reddit and return the results."""
        search = SerpAdvancedGoogleSearch(
            site="reddit.com",
            language=self.tool_manager.language,
            region=self.tool_manager.region,
            date_filter=self.tool_manager.date_filter,
        )
        result = await search.run(query)
        if not result or  result == 'Could not search Google. Please try again later.':
            return []
        result = search.process_response(result)
        return result

    async def send_event(self, send, response_streamer, data):
        if not data:
            return

        search_results_response_body = {
            "type": "reddit_search",
            "content": data,
        }

        response_streamer.more_body = False

        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(search_results_response_body).encode("utf-8"),
                "more_body": False,
            }
        )

        bt.logging.info("Reddit search results data sent")
