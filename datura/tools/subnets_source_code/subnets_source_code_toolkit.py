from abc import ABC
from typing import List
from datura.tools.base import BaseToolkit, BaseTool
from datura.tools.subnets_source_code.subnets_source_code_summary import summarize_subnet_source_code_data
from datura.tools.subnets_source_code.subnets_source_code_tool import SubnetsSourceCodeTool


class SubnetsSourceCodeToolkit(BaseToolkit, ABC):
    name: str = "Subnets Source Code Toolkit"
    description: str = "Toolkit containing tools for interacting with Bittensor Subnets' source code."
    slug: str = "subnets-source-code"
    toolkit_id = "f4f14493-79a4-430a-9eab-f6c0e9d9e369"

    def get_tools(self) -> List[BaseTool]:
        return [SubnetsSourceCodeTool()]

    async def summarize(self, prompt, model, data):
        response_order = self.tool_manager.response_order
        data = next(iter(data.values()))
        return await summarize_subnet_source_code_data(
            prompt=prompt,
            model=model,
            docs=data,
            response_order=response_order,
        )
