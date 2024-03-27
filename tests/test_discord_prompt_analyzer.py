import asyncio
from template.dataset.dataset import MockDiscordQuestionsDataset
from template.services.discord_prompt_analyzer import DiscordPromptAnalyzer
import bittensor as bt


async def main():
    client = DiscordPromptAnalyzer()
    dt = MockDiscordQuestionsDataset()

    for _ in range(len(dt.question_templates)):
        prompt = dt.next()
        result = await client.analyse_prompt_and_fetch_messages(prompt)
        bt.logging.warning(f"{result}")


if __name__ == "__main__":
    asyncio.run(main())
