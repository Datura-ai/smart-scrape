import os
import json
from typing import List
from pinecone import Pinecone
import asyncio
from llama_index.core.storage.index_store.utils import json_to_index_struct
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.schema import NodeWithScore

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


class PineconeIndexer:
    def __init__(self) -> None:
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        self.channels = self.read_channels()
        self.router = self.get_router_retriever()

    def read_channels(self) -> dict:
        """
        Returns all channel indexes of pinecoine with name and
        description match in dict.
        """

        file_path = os.path.join(os.path.dirname(__file__), "pinecone_indexes.json")
        with open(file_path, "r") as file:
            return json.load(file)

    def create_json_to_index_struct(self, index: str):
        return json_to_index_struct(
            {"__type__": "vector_store",
                "__data__": f'{{"index_id": "{index}", "...dict": {{}}}}'}
        )

    def get_retriever_tools(
        self,
        similarity_top_k: int
    ) -> List[RetrieverTool]:
        retriever_tools = []

        for index, ntd in self.channels.items():
            for namespace, description in ntd.items():
                try:
                    vector_store = PineconeVectorStore(
                        pinecone_index=self.pc.Index(index),
                        api_key=PINECONE_API_KEY,
                        namespace=namespace,
                    )
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store,
                    )
                    vector_store_index = VectorStoreIndex(
                        index_struct=self.create_json_to_index_struct(index),
                        storage_context=storage_context,
                        embed_model=self.embed_model,
                    )
                    retriever_tool = RetrieverTool.from_defaults(
                        retriever=vector_store_index.as_retriever(
                            similarity_top_k=similarity_top_k,
                        ),
                        description=description,
                        name=namespace
                    )
                    retriever_tools.append(retriever_tool)
                except Exception as e:
                    print(
                        f"Failed to initialize retriever for index {index} | namespace {namespace}: {e}")

        return retriever_tools

    def get_router_retriever(self, similarity_top_k=5) -> RouterRetriever:
        retriever_tools = self.get_retriever_tools(similarity_top_k)
        return RouterRetriever.from_defaults(
            retriever_tools=retriever_tools,
            select_multi=True,
        )

    async def general_retrieve(
        self,
        query: str,
        similarity_top_k: int,
        index_names: List[str],
    ):
        if not index_names:
            return await self.retrieve(query, similarity_top_k)

        docs = await self.retrieve_with_index_names(
            query,
            similarity_top_k,
            index_names
        )

        if len(docs) > 0:
            return docs

        return await self.retrieve(query, similarity_top_k)

    async def retrieve(
        self,
        query: str,
        similarity_top_k: int,
    ):
        router = self.get_router_retriever()
        all_results = await router.aretrieve(query)
        all_results.sort(key=lambda x: x.score, reverse=True)
        if similarity_top_k > len(all_results):
            return self.nodes_to_json(all_results)

        return self.nodes_to_json(all_results[:similarity_top_k])

    async def retrieve_with_index_names(
        self, query: str, similarity_top_k: int, index_names: List[str]
    ):
        retriever_tools = self.get_retriever_tools(similarity_top_k)

        async def retrieve_from_index(index_name):
            for retriever_tool in retriever_tools:
                if retriever_tool.metadata.name == index_name:
                    return await retriever_tool.retriever.aretrieve(query)
            return []

        tasks = [retrieve_from_index(index_name) for index_name in index_names]
        results = await asyncio.gather(*tasks)

        all_results = []
        for result in results:
            all_results.extend(result)

        all_results.sort(key=lambda x: x.score, reverse=True)

        if similarity_top_k > len(all_results):
            return self.nodes_to_json(all_results)

        return self.nodes_to_json(all_results[:similarity_top_k])

    def nodes_to_json(
        self,
        nodes: List[NodeWithScore]
    ) -> List[dict]:
        data = []
        for node in nodes:
            data.append({'content': node.node.text, 'link': node.node.metadata.get('link')})

        return data
