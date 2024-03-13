import json
import bittensor as bt
from typing import Type
from starlette.types import Send
from pydantic import BaseModel, Field
from template.services.twitter_prompt_analyzer import TwitterPromptAnalyzer
from template.tools.base import BaseTool


class GetFullArchiveTweetsSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Twitter",
    )


class GetFullArchiveTweetsTool(BaseTool):
    """Tool that gets full tweet archive from Twitter."""

    name = "Full Archive Tweets"

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
        openai_query_model = self.tool_manager.miner.config.miner.openai_query_model
        openai_fix_query_model = (
            self.tool_manager.miner.config.miner.openai_fix_query_model
        )

        client = TwitterPromptAnalyzer(
            openai_query_model=openai_query_model,
            openai_fix_query_model=openai_fix_query_model,
        )

        result, prompt_analysis = await client.analyse_prompt_and_fetch_tweets(
            query, is_recent_tweets=False
        )

        bt.logging.info(
            "================================== Prompt analysis ==================================="
        )
        bt.logging.info(prompt_analysis)
        bt.logging.info(
            "================================== Prompt analysis ===================================="
        )

        if self.tool_manager:
            self.tool_manager.twitter_data = result
            self.tool_manager.twitter_prompt_analysis = prompt_analysis

        return (result, prompt_analysis)

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        tweets, prompt_analysis = data

        # Send prompt_analysis
        if prompt_analysis:
            prompt_analysis_response_body = {
                "type": "prompt_analysis",
                "content": prompt_analysis.dict(),
            }

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(prompt_analysis_response_body).encode("utf-8"),
                    "more_body": True,
                }
            )
            bt.logging.info("Prompt Analysis sent")

        if tweets:
            tweets_amount = tweets.get("meta", {}).get("result_count", 0)

            tweets_response_body = {"type": "tweets", "content": tweets}
            response_streamer.more_body = False

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(tweets_response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info(f"Tweet data sent. Number of tweets: {tweets_amount}")
