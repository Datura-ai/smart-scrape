from typing import Type

from pydantic import BaseModel, Field

from template.tools.base import BaseTool

from template.services.twitter_api_wrapper import TwitterAPIClient


class GetRecentTweetsToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for tweets.",
    )


class GetRecentTweetsTool(BaseTool):
    """Tool that gets recent tweets from Twitter."""

    name = "Get Recent Tweets"

    slug = "get_recent_tweets"

    description = "Get recent tweets for a given query."

    args_schema: Type[GetRecentTweetsToolSchema] = GetRecentTweetsToolSchema

    tool_id = "6e57b718-8953-448b-98db-fd19c1d1469c"

    def _run():
        pass

    async def _arun(
        self,
        query: str,  # run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Tweet message and return."""
        client = TwitterAPIClient()
        result = await client.analyse_prompt_and_fetch_tweets(query)
        return result
