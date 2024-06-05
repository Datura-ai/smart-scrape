from abc import ABC
from typing import List
from datura.tools.base import BaseToolkit, BaseTool
from datura.tools.bittensor.bittensor_summary import summarize_bittensor_data
from datura.tools.bittensor.bittensor_docs_tool import BittensorDocsTool


class BittensorToolkit(BaseToolkit, ABC):
    name: str = "Bittensor Toolkit"
    description: str = "Toolkit containing tools for interacting with Bittensor."
    slug: str = "bittensor"
    toolkit_id = "f4f14493-79a4-430a-9eab-f6c0e9d9e369"

    def get_tools(self) -> List[BaseTool]:
        return [BittensorDocsTool()]

    async def summarize(self, prompt, model, data):
        response_order = self.tool_manager.response_order
        data = next(iter(data.values()))
        return await summarize_bittensor_data(
            prompt=prompt,
            model=model,
            docs=data,
            response_order=response_order
        )
