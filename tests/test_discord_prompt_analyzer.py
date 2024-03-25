import asyncio
from template.dataset.dataset import MockDiscordQuestionsDataset
from template.services.discord_prompt_analyzer import DiscordPromptAnalyzer
import bittensor as bt


async def main():
    client = DiscordPromptAnalyzer()
    dt = MockDiscordQuestionsDataset()

    for _ in range(len(dt.question_templates)):
        prompt = dt.next()
        _, prompt_analysis = await client.generate_and_analyze_query(prompt)

        bt.logging.info("================================== Prompt analysis ==================================")
        bt.logging.info(prompt)
        bt.logging.info(prompt_analysis)
        bt.logging.info("================================== Prompt analysis ==================================")


if __name__ == "__main__":
    asyncio.run(main())
