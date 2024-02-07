from typing import Type

from pydantic import BaseModel, Field
from template.services.twitter_api_wrapper import TwitterAPIClient
from template.tools.base import BaseTool


class GetFullArchiveTweetsSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Twitter",
    )


class GetFullArchiveTweetsTool(BaseTool):
    """Tool that gets full tweet archive from Twitter."""

    name = "Get Full Tweets Archive"

    slug = "get_full_archive_tweets"

    description = "Get full archive tweets for a given query."

    args_schema: Type[GetFullArchiveTweetsSchema] = GetFullArchiveTweetsSchema

    tool_id = "319ae0a3-feeb-46a9-8c2c-5a955a17c854"

    def _run():
        pass

    async def _arun(
        self,
        query: str,  # run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        client = TwitterAPIClient()
        result = await client.analyse_prompt_and_fetch_tweets(
            query, is_recent_tweets=False
        )
        return result
