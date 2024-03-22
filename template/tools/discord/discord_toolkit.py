from abc import ABC
from typing import List
from template.tools.base import BaseToolkit, BaseTool
from template.tools.discord.discord_summary import prepare_messages_data_for_summary, summarize_discord_data
from template.tools.discord.search_tool import SearchDiscordTool


class DiscordToolkit(BaseToolkit, ABC):
    name: str = "Discord Toolkit"
    description: str = "Toolkit containing tools for interacting discord."
    slug: str = "discord"
    toolkit_id = "fb78b028-f7f4-4d20-b7e8-7dc072e97d9a"

    def get_tools(self) -> List[BaseTool]:
        return [SearchDiscordTool()]

    async def summarize(self, prompt, model, data):
        messages, prompt_analysis = data

        return await summarize_discord_data(
            prompt=prompt,
            model=model,
            filtered_tweets=prepare_messages_data_for_summary(messages),
            prompt_analysis=prompt_analysis,
        )
