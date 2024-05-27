import traceback
import bittensor as bt
from starlette.types import Send
from datura.protocol import (
    ScraperStreamingSynapse,
)
from datura.tools.tool_manager import ToolManager
from datura.dataset.date_filters import DateFilter, DateFilterType
from datetime import datetime
import pytz


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
            response_order = synapse.response_order
            response_size = synapse.response_size
            response_type = synapse.response_type

            bt.logging.trace(synapse)

            bt.logging.info(
                "================================== Prompt ==================================="
            )
            bt.logging.info(prompt)
            bt.logging.info(
                "================================== Prompt ===================================="
            )
            start_date = datetime.strptime(
                synapse.start_date, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=pytz.utc)
            end_date = datetime.strptime(
                synapse.end_date, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=pytz.utc)

            date_filter = DateFilter(
                start_date=start_date,
                end_date=end_date,
                date_filter_type=DateFilterType(synapse.date_filter_type),
            )

            tool_manager = ToolManager(
                prompt=prompt,
                manual_tool_names=tools,
                send=send,
                is_intro_text=is_intro_text,
                miner=self.miner,
                language=synapse.language,
                region=synapse.region,
                date_filter=date_filter,
                google_date_filter=synapse.google_date_filter,
                response_order=response_order.value,
                response_size=response_size,
                response_type=response_type.value,
            )

            await tool_manager.run()

            bt.logging.info("End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")
