import bittensor as bt
from datura.protocol import WebSearchSynapse


class WebSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def search(self, synapse: WebSearchSynapse):
        # Extract the query from the synapse
        query = synapse.query

        # Log the mock search execution
        bt.logging.info(f"Executing mock web search with query: {query}")

        # Mock result
        mock_result = {
            "title": "Mock Search Result Title",
            "snippet": "This is a mock snippet for the search result.",
            "link": "https://example.com/mock-result",
            "date": "1 hour ago",
            "source": "Mock Source",
            "author": "Mock Author",
            "image": "https://example.com/mock-image.jpg",
            "favicon": "https://example.com/mock-favicon.ico",
            "highlights": ["This is a highlighted text from the mock result."],
        }

        # Assign the mock result to the results field of the synapse
        synapse.results = {"data": [mock_result]}

        return synapse
