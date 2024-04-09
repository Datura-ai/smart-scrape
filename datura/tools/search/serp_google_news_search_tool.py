import os
import json
import bittensor as bt
from typing import Type, Optional
from pydantic import BaseModel, Field
from datura.tools.base import BaseTool
from starlette.types import Send
from .serp_api_wrapper import SerpAPIWrapper


SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md"
    )


class SerpGoogleNewsSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Google News search.",
    )


class SerpGoogleNewsSearchTool(BaseTool):
    search: Optional[SerpAPIWrapper] = None

    def __init__(self):
        super().__init__()
        self.search = SerpAPIWrapper(
            serpapi_api_key=SERPAPI_API_KEY, params={"engine": "google", "tbm": "nws"}
        )

    name = "Google News Search"

    slug = "serp_google_news_search"

    description = (
        "This tool performs Google searches and extracts relevant snippets and webpages. "
        "It's particularly useful for staying updated with current events and finding quick answers to your queries."
    )

    args_schema: Type[SerpGoogleNewsSearchSchema] = SerpGoogleNewsSearchSchema

    tool_id = "d3f5c303-e2a4-4cde-8a6b-cf5b2b6f1204"

    def _run():
        pass

    async def _arun(
        self,
        query: str,
    ):
        """Search Google News and return the results."""
        try:
            return await self.search.arun(query)
        except Exception as err:
            if "Invalid API key" in str(err):
                bt.logging.error(f"SERP API Key is invalid: {err}")
                return "SERP API Key is invalid"

            bt.logging.error(f"Could not perform SERP Google Search: {err}")
            return "Could not search Google. Please try again later."

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        search_results_response_body = {
            "type": "google_search_news",
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

        bt.logging.info("Web search results data sent")
