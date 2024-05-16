import re
import json
import bittensor as bt
from typing import Type
from datura.tools.subnets_source_code.pinecone_indexer import SubnetSourceCodePineconeIndexer
from pydantic import BaseModel, Field
from starlette.types import Send
from datura.tools.base import BaseTool


class SubnetsSourceCodeToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The question for subnets source code.",
    )


class SubnetsSourceCodeTool(BaseTool):
    """Tool that answers questions about Bittensor docs."""

    name = "Subnet Source Code"
    slug = "subnets-source-code"
    description = "Answer questions about bittensor subnet's source code."
    args_scheme: Type[SubnetsSourceCodeToolSchema] = SubnetsSourceCodeToolSchema
    tool_id = "98590eca-7db9-495f-9d35-c7c1aeeffe0b"

    def _run():
        pass

    def extract_channels_from_query(self, query: str):
        channels = []

        # Pattern 1: #\[channel\](channel)
        pattern1 = r"#\[(\w+(-\w+)*)\]\(\w+(-\w+)*\)"
        matches1 = re.findall(pattern1, query)
        channels.extend([match[0] for match in matches1])

        # Pattern 2: #channel
        pattern2 = r"#(\w+(?:-\w+)*)"
        matches2 = re.findall(pattern2, query)
        channels.extend(matches2)

        # Remove channel names from the query text
        query = re.sub(pattern1, "", query)
        query = re.sub(pattern2, "", query)

        # Remove extra whitespace from the query text
        query = re.sub(r"\s+", " ", query).strip()
        return channels, query

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Search on subnets' source codes"""
        client = SubnetSourceCodePineconeIndexer()

        limit = 15
        index_names, query = self.extract_channels_from_query(query)

        docs = await client.general_retrieve(query, limit, index_names)

        bt.logging.info(
            "================================== Subnets Source Code Result ==================================="
        )
        bt.logging.info(docs)
        bt.logging.info(
            "================================== Subnets Source Code Result ===================================="
        )

        return docs

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        if data:
            response_body = {
                "type": "subnets_source_code_search",
                "content": data,
            }

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info("Subnets Source Code search results data sent")
