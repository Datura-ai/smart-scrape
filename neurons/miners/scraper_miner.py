import traceback
import bittensor as bt
from starlette.types import Send
from datura.protocol import (
    ScraperStreamingSynapse,
)
from datura.tools.tool_manager import ToolManager


class ScraperMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def smart_scraper(self, synapse: ScraperStreamingSynapse, send: Send):
        try:
            model = synapse.model
            prompt = synapse.messages
            seed = synapse.seed
            tools = synapse.tools
            is_intro_text = synapse.is_intro_text

            bt.logging.trace(synapse)

            bt.logging.info(
                "================================== Prompt ==================================="
            )
            bt.logging.info(prompt)
            bt.logging.info(
                "================================== Prompt ===================================="
            )

            tool_manager = ToolManager(
                prompt=prompt,
                manual_tool_names=tools,
                send=send,
                is_intro_text=is_intro_text,
                miner=self.miner,
                language=synapse.language,
                region=synapse.region,
                date_filter=synapse.date_filter,
            )

            await tool_manager.run()

            bt.logging.info("End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")
