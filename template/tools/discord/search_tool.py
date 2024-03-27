import json
from typing import Type
from pydantic import BaseModel, Field
from starlette.types import Send
from template.services.discord_prompt_analyzer import DiscordPromptAnalyzer
from template.tools.base import BaseTool
import bittensor as bt


class DiscordSearchToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Discord messages.",
    )


class DiscordSearchTool(BaseTool):
    """Tool that searches for messages in Discord based on a query."""

    name = "Discord Search"
    slug = "search_discord"
    description = "Search for messages in Discord for a given query."
    args_scheme: Type[DiscordSearchToolSchema] = DiscordSearchToolSchema
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

        result, discord_prompt_analysis = await client.analyse_prompt_and_fetch_messages(query)
        bt.logging.info(
            "================================== Discord Prompt analysis ==================================="
        )
        bt.logging.info(discord_prompt_analysis)
        bt.logging.info(
            "================================== Discord Prompt analysis ===================================="
        )

        return result, discord_prompt_analysis

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        messages, discord_prompt_analysis = data
        # Send prompt_analysis
        if discord_prompt_analysis:
            discord_prompt_analysis_response_body = {
                "type": "discord_prompt_analysis",
                "content": discord_prompt_analysis.dict(),
            }

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(discord_prompt_analysis_response_body).encode("utf-8"),
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
