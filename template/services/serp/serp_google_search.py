from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.utilities.serpapi import SerpAPIWrapper
from pydantic import BaseModel, Field

from exceptions import ToolEnvKeyException
from tools.base import BaseTool


class SerpGoogleSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Google search.",
    )


class SerpGoogleSearchTool(BaseTool):
    name = "Serp Google Search"

    slug = "serpGoogleSearch"

    description = (
        "This tool performs Google searches and extracts relevant snippets and webpages. "
        "It's particularly useful for staying updated with current events and finding quick answers to your queries."
    )

    args_schema: Type[SerpGoogleSearchSchema] = SerpGoogleSearchSchema

    tool_id = "a66b3b20-d0a2-4b53-a775-197bc492e816"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search Google and return the results."""
        serpapi_api_key = self.get_env_key("SERP_API_KEY")

        if not serpapi_api_key:
            raise ToolEnvKeyException(
                f"Please fill Serp API Key in the [Google SERP Search Toolkit](/toolkits/{self.toolkit_slug})"
            )

        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

        try:
            return search.run(query)
        except Exception as err:
            if "Invalid API key" in str(err):
                raise ToolEnvKeyException(
                    f"Serp API Key is not valid. Please check in the [Google SERP Search Toolkit](/toolkits/{self.toolkit_slug})"
                )

            return "Could not search Google. Please try again later."
