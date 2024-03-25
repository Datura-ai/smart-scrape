import os
import aiohttp
import bittensor as bt
from typing import Optional


BASE_URL = "http://daturadiscord.eu-north-1.elasticbeanstalk.com"


class DiscordAPIClient:
    def __init__(
        self,
        openai_query_model="gpt-3.5-turbo-0125",
        openai_fix_query_model="gpt-4-1106-preview",
    ):
        self.url = BASE_URL,
        self.headers = {
            "Content-Type": "application/json",
        }
        self.openai_query_model = openai_query_model
        self.openai_fix_query_model = openai_fix_query_model

    async def connect_to_endpoint(self, url, params: Optional[str] = None, body: Optional[str] = None):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, json=body) as response:
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

    async def get_all_channels(self, query):
        """
        Retrieve all channels for a specified guild.
        :return: A list of channel objects.
        """
        url = f"{self.url}/api/channels"
        return await self.connect_to_endpoint(url, query)

    async def search_messages(self, body):
        """
        Get messages from a channel and then fetch full details for each.
        :param body: containing, limit, page and query.

        :return: A list of messages
        """
        url = f"{self.url}/api/messages"
        return await self.connect_to_endpoint(url, body)
