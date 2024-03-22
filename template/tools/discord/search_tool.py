import json
from typing import Type
from pydantic import BaseModel, Field
from starlette.types import Send
from template.services.discord_prompt_analyzer import DiscordPromptAnalyzer
from template.tools.base import BaseTool
import bittensor as bt


class SearchDiscordToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Discord messages.",
    )


class SearchDiscordTool(BaseTool):
    """Tool that searches for messages in Discord based on a query."""

    name = "Search Discord"
    slug = "search_discord"
    description = "Search for messages in Discord for a given query."
    args_scheme: Type[SearchDiscordToolSchema] = SearchDiscordToolSchema
    tool_id = "dd29715f-066f-4f8d-8adb-2dd005380f03"

    def _run():
        pass

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Search Discord messages and return results."""
        openai_query_model = self.tool_manager.miner.config.miner.openai_query_model
        openai_fix_query_model = (
            self.tool_manager.miner.config.miner.openai_fix_query_model
        )

        client = DiscordPromptAnalyzer(
            openai_query_model=openai_query_model,
            openai_fix_query_model=openai_fix_query_model,
        )

        result, prompt_analysis = await client.analyse_prompt_and_fetch_messages(query)
        bt.logging.info(
            "================================== Prompt analysis ==================================="
        )
        bt.logging.info(prompt_analysis)
        bt.logging.info(
            "================================== Prompt analysis ===================================="
        )

        return (result, prompt_analysis)

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        messages, prompt_analysis = data
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

        if messages:
            # We may need more body here to continueslly fetch messages
            messages_response_body = {"type": "discord_messages", "content": messages}

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(messages_response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info(f"Discord search data sent")
