import os
import aiohttp
import bittensor as bt
from typing import Optional

BASE_URL = "http://api-discord.datura.ai"

class DiscordAPIClient:
    def __init__(
        self,
    ):
        self.headers = {
            "Content-Type": "application/json",
        }

    async def connect_to_endpoint(self, url, body: Optional[str] = None):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, json=body) as response:
                if response.status in [401, 403]:
                    bt.logging.error(
                        f"Critical Discord API Request error occurred: {await response.text()}"
                    )
                    os._exit(1)

                json_data = None

                try:
                    json_data = await response.json()
                except aiohttp.ContentTypeError:
                    pass

                response_text = await response.text()
                return json_data, response.status, response_text

    async def search_messages(self, body):
        """
        Get messages from a channel and then fetch full details for each.
        :param body: containing, limit, page and query.

        :return: A list of messages
        """
        url = f"{BASE_URL}/api/messages"
        return await self.connect_to_endpoint(url, body)
