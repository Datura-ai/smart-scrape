import re
import json
import bittensor as bt
from typing import Type
from datura.tools.bittensor.pinecone_indexer import PineconeIndexer
from pydantic import BaseModel, Field
from starlette.types import Send
from datura.tools.base import BaseTool


class BittensorDocsToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="The question for Bittensor docs.",
    )


class BittensorDocsTool(BaseTool):
    """Tool that answers questions about Bittensor docs."""

    name = "Bittensor Docs"
    slug = "bittensor-docs"
    description = "Answer questions about Bittensor docs."
    args_scheme: Type[BittensorDocsToolSchema] = BittensorDocsToolSchema
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
        """Search Bittensor Documentation and return results."""
        client = PineconeIndexer()

        limit = 15
        index_names, query = self.extract_channels_from_query(query)

        docs = await client.general_retrieve(query, limit, index_names)

        bt.logging.info(
            "================================== Bittensor Docs Result ==================================="
        )
        bt.logging.info(docs)
        bt.logging.info(
            "================================== Bittensor Docs Result ===================================="
        )

        return docs

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        if data:
            response_body = {
                "type": "bittensor_docs_search",
                "content": data,
            }

            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info("Bittensor Documentation search results data sent")
