import os
import json
import bittensor as bt
from typing import Type
from pydantic import BaseModel, Field
from datura.tools.base import BaseTool
from starlette.types import Send
from datura.tools.search.serp_api_wrapper import SerpAPIWrapper


SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md"
    )


class SerpGoogleImageSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Google search.",
    )


class SerpGoogleImageSearchTool(BaseTool):
    name = "Google Image Search"

    slug = "serp_google_image_search"

    description = "This tool performs Google image search"

    args_schema: Type[SerpGoogleImageSearchSchema] = SerpGoogleImageSearchSchema

    tool_id = "a66b3b20-d0a2-4b53-a775-197bc492e816"

    def _run():
        pass

    async def _arun(
        self,
        query: str,
    ):
        """Search Google images and return the results."""

        search = SerpAPIWrapper(
            serpapi_api_key=SERPAPI_API_KEY, params={"engine": "google_images"}
        )

        try:
            return await search.arun(query)
        except Exception as err:
            if "Invalid API key" in str(err):
                bt.logging.error(f"SERP API Key is invalid: {err}")
                return "SERP API Key is invalid"

            bt.logging.error(f"Could not perform SERP Google Image Search: {err}")
            return "Could not search Google images. Please try again later."

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        image_search_results_response_body = {
            "type": "google_image_search",
            "content": data,
        }

        response_streamer.more_body = False

        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(image_search_results_response_body).encode("utf-8"),
                "more_body": False,
            }
        )

        bt.logging.info("Google image search results data sent")
