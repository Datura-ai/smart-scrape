import json
import bittensor as bt
from typing import Type
from pydantic import BaseModel, Field
from starlette.types import Send
from datura.services.discord_messages.discord_service import DiscordService
from datura.tools.base import BaseTool


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
        date_filter = self.tool_manager.date_filter

        client = DiscordService()

        result = await client.search(
            query=query,
            limit=8,
            possible_reply_limit=8,
            start_date=date_filter.start_date.timestamp(),
            end_date=date_filter.end_date.timestamp(),
        )
        bt.logging.info(
            "================================== Discord Result ==================================="
        )
        bt.logging.info(result)
        bt.logging.info(
            "================================== Discord Result ===================================="
        )

        return result

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        if data:
            messages_response_body = {
                "type": "discord_search",
                "content": data,
            }

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(messages_response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info("Discord search results data sent")
