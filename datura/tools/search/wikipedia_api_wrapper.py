from langchain_community.utilities import (
    WikipediaAPIWrapper as LangchainWikipediaAPIWrapper,
)
from wikipedia import set_rate_limiting
import bittensor as bt
from datetime import timedelta

WIKIPEDIA_MAX_QUERY_LENGTH = 300
MAX_DOC_CONTENT_CHARS = 1000

set_rate_limiting(True, min_wait=timedelta(milliseconds=50))


class WikipediaAPIWrapper(LangchainWikipediaAPIWrapper):
    def run(self, query: str) -> str:
        try:
            page_titles = self.wiki_client.search(
                query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results
            )
            summaries = []
            for page_title in page_titles[: self.top_k_results]:
                if wiki_page := self._fetch_page(page_title):
                    summaries.append(
                        {
                            "title": page_title,
                            "summary": wiki_page.summary[:MAX_DOC_CONTENT_CHARS],
                            "url": wiki_page.url,
                        }
                    )
            if not summaries:
                bt.logging.info("No good Wikipedia Search Result was found")

            return summaries
        except Exception as e:
            bt.logging.error(f"Error occurred while fetching Wikipedia results: {e}")
            return []
