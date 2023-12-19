
from miner.dataset_twitter.mock import get_random_tweets
from elasticsearch import Elasticsearch, NotFoundError, RequestError
from datetime import datetime, timedelta
import asyncio
import random
import json

ELASTIC_HOST = 'localhost'
ELASTIC_USER = 'elastic'
ELASTIC_PORT = 9200
ELASTIC_PASSWORD = 'ORHO=WmV7TfdAKVb+VEY'
example_prompts = [
    'Gather opinions on the new iPhone model from tech experts on Twitter.',
    'Find tweets about climate change from the last year.',
    'Show me the latest tweets about the SpaceX launch.',
    'Collect tweets reacting to the latest UN summit.',
    "Last year's trends  about #openai",
    "Tell me last news about elonmusk"
]

from template.utils import get_version, analyze_twitter_query
from template.protocol import StreamPrompting, IsAlive, TwitterScraper, TwitterQueryResult

class OpenSearchClient:
    def __init__(self, index = 'tweets'):
        self.es = Elasticsearch(
            hosts=[{'host': ELASTIC_HOST, 'port': ELASTIC_PORT, 'scheme': 'https'}],
            basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
            verify_certs=False  # This is equivalent to the -k flag in curl
        )
        self.index = 'tweets' if index is None else index

    def create_or_update_document(self, doc_id, document):
        try:
            # Update if exists, else create a new document
            return self.es.update(index=self.index, id=doc_id, body={'doc': document, 'doc_as_upsert': True})
        except RequestError as e:
            print(f"Error: {e}")
            return None

    def find_document(self, doc_id):
        try:
            return self.es.get(index=self.index, id=doc_id)
        except NotFoundError:
            print(f"Document ID {doc_id} not found.")
            return None

    def search_documents(self, query):
        try:
        
            if isinstance(query, dict):
                return self.es.search(index=self.index, body=query)
            else:
                print("Query is not a valid JSON object.")
                return None
        except RequestError as e:
            print(f"Search Error: {e}")
            return None
        
    async def perform_twitter_query_and_search(self, prompt):
        try:
            query_result: TwitterQueryResult = await analyze_twitter_query(prompt)
            print("Prompt ==================================")
            print(prompt)
            print("Query-Start ==================================")
            print(query_result.query_string)
            print("Query-End  ==================================")

            # Check if the query string is not empty and is a valid JSON string
            if isinstance(query_result.query_string, str) and query_result.query_string.strip():
                try:
                    query_dict = json.loads(query_result.query_string)
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    return None
            else:
                query_dict = query_result.query_string

            # Assuming analyze_twitter_query returns a query suitable for search_documents
            search_results = self.search_documents(query_dict)
            print("Start-Search-Result ===================================")
            print(search_results)
            print("End-Search-Result ===================================")
        except RequestError as e:
            print(f"Search Error: {e}")
            return None



# Usage Example
if __name__ == "__main__":
    # Replace with your OpenSearch details

    client = OpenSearchClient()


    # tweets = get_random_tweets(2000)
    # for tweet in tweets:
    #     result = client.create_or_update_document( tweet['id'], tweet)
    #     print(result)
    

    # # Find a document
    # found_doc = client.find_document('your-index-name', 'doc-id-123')
    # print(found_doc)


    # random_index = random.randint(0, len(example_prompts) - 1)  # Make sure to import random

    for pr in example_prompts:
        # Call the new method
        asyncio.run(client.perform_twitter_query_and_search(pr))
