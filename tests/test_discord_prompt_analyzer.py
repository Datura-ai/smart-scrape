import asyncio
from datura.dataset.dataset import MockDiscordQuestionsDataset
from datura.services.discord_prompt_analyzer import DiscordPromptAnalyzer
import bittensor as bt

from datura.tools.discord.discord_summary import summarize_discord_data


async def main():
    client = DiscordPromptAnalyzer()
    dt = MockDiscordQuestionsDataset()

    for _ in range(len(dt.question_templates)):
        prompt = dt.next()
        result_json, prompt_analysis = await client.analyse_prompt_and_fetch_messages(
            prompt
        )
        res, role = await summarize_discord_data(
            prompt, "gpt-3.5-turbo-0125", result_json, prompt_analysis
        )
        bt.logging.info("===================================================")
        bt.logging.info(f"Messages {result_json}")
        bt.logging.info("===================================================")
        bt.logging.info(f"Role {role} \n Result {res}")
        bt.logging.info("===================================================")


if __name__ == "__main__":
    asyncio.run(main())
