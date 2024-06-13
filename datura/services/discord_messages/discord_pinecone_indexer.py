import os
import json
import asyncio
from pydantic import Field
from pinecone import Pinecone
from llama_index.core.tools import RetrieverTool
from llama_index.core.schema import NodeWithScore
from typing import Any, Callable, Dict, List, Optional
from llama_index.core.retrievers import RouterRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.storage.index_store.utils import json_to_index_struct
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
    MetadataFilter,
    FilterOperator
)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


class RecencyPineconeVectorStore(PineconeVectorStore):
    start_date: Optional[int] = Field(default=None)
    end_date: Optional[int] = Field(default=None)

    def __init__(
        self,
        pinecone_index: Optional[Any] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: Optional[str] = None,
        insert_kwargs: Optional[Dict] = None,
        add_sparse_vector: bool = False,
        tokenizer: Optional[Callable] = None,
        text_key: str = "text",
        batch_size: int = 100,
        remove_text_from_metadata: bool = False,
        default_empty_query_vector: Optional[List[float]] = None,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pinecone_index=pinecone_index,
            api_key=api_key,
            index_name=index_name,
            environment=environment,
            namespace=namespace,
            insert_kwargs=insert_kwargs,
            add_sparse_vector=add_sparse_vector,
            tokenizer=tokenizer,
            text_key=text_key,
            batch_size=batch_size,
            remove_text_from_metadata=remove_text_from_metadata,
            default_empty_query_vector=default_empty_query_vector,
            **kwargs,
        )
        self.start_date = start_date
        self.end_date = end_date

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        recency_filter = self._create_recency_filter()
        query.filters = recency_filter
        return super().query(query, **kwargs)

    def _create_recency_filter(self) -> MetadataFilters:
        start = self.start_date
        end = self.end_date
        return MetadataFilters(filters=[
            MetadataFilter(
                key='timestamp_date',
                value=start,
                operator=FilterOperator.GTE,
            ),
            MetadataFilter(
                key='timestamp_date',
                value=end,
                operator=FilterOperator.LTE,
            )
        ])


class PineconeIndexer:
    def __init__(self) -> None:
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.channels = self.read_channels()

    def read_channels(self) -> dict:
        """
        Returns all channel indexes of pinecoine with name and
        description match in dict.
        """

        file_path = os.path.join(os.path.dirname(
            __file__), "pinecone_indexes.json")
        with open(file_path, "r") as file:
            return json.load(file)

    def create_json_to_index_struct(self, index: str):
        return json_to_index_struct(
            {
                "__type__": "vector_store",
                "__data__": f'{{"index_id": "{index}", "...dict": {{}}}}',
            }
        )

    def get_retriever_tools(
        self,
        similarity_top_k: int,
        start_date: Optional[int],
        end_date: Optional[int],
    ) -> List[RetrieverTool]:
        retriever_tools = []

        for index, ntd in self.channels.items():
            for namespace, description in ntd.items():
                try:
                    vector_store = RecencyPineconeVectorStore(
                        pinecone_index=self.pc.Index(index),
                        api_key=PINECONE_API_KEY,
                        namespace=namespace,
                        start_date=start_date,
                        end_date=end_date,
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
                        name=namespace,
                        description=description,
                    )
                    retriever_tools.append(retriever_tool)
                except Exception as e:
                    print(f"Failed to initialize retriever for index {index} | namespace {namespace}: {e}")

        return retriever_tools

    def get_router_retriever(
        self,
        similarity_top_k: int,
        start_date: Optional[int],
        end_date: Optional[int],
    ) -> RouterRetriever:
        retriever_tools = self.get_retriever_tools(
            similarity_top_k,
            start_date,
            end_date,
        )
        return RouterRetriever.from_defaults(
            retriever_tools=retriever_tools,
            select_multi=True,
        )

    async def retrieve(
        self,
        query: str,
        similarity_top_k: int,
        start_date: int,
        end_date: int,
    ) -> List[NodeWithScore]:
        router = self.get_router_retriever(
            similarity_top_k,
            start_date,
            end_date,
        )
        all_results = await router.aretrieve(query)
        all_results.sort(key=lambda x: x.score, reverse=True)

        if similarity_top_k > len(all_results):
            return all_results

        return all_results[:similarity_top_k]

    async def retrieve_with_index_names(
        self,
        query: str,
        similarity_top_k: int,
        index_names: List[str],
        start_date: Optional[int],
        end_date: Optional[int],
    ):
        retriever_tools = self.get_retriever_tools(
            similarity_top_k,
            start_date,
            end_date,
        )

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
            return all_results

        return all_results[:similarity_top_k]
