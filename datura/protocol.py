import pydantic
import bittensor as bt
import typing
import json
import asyncio
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Awaitable, Dict, Optional, Any
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from enum import Enum
from aiohttp import ClientResponse
from datura.services.twitter_utils import TwitterUtils
from datura.services.web_search_utils import WebSearchUtils
import traceback


class IsAlive(bt.Synapse):
    answer: typing.Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )


class TwitterPromptAnalysisResult(BaseModel):
    api_params: Dict[str, Any] = {}
    keywords: List[str] = []
    hashtags: List[str] = []
    user_mentions: List[str] = []

    def fill(self, response: Dict[str, Any]):
        if "api_params" in response:
            self.api_params = response["api_params"]
        if "keywords" in response:
            self.keywords = response["keywords"]
        if "hashtags" in response:
            self.hashtags = response["hashtags"]
        if "user_mentions" in response:
            self.user_mentions = response["user_mentions"]

    def __str__(self):
        return f"Query String: {self.api_params}, Keywords: {self.keywords}, Hashtags: {self.hashtags}, User Mentions: {self.user_mentions}"


class TwitterScraperMedia(BaseModel):
    media_url: str = ""
    type: str = ""


class TwitterScraperUser(BaseModel):
    # Available in both, scraped and api based tweets.
    id: Optional[str]
    url: Optional[str]
    name: Optional[str]
    username: Optional[str]
    created_at: Optional[str]

    # Only available in scraped tweets
    description: Optional[str]
    favourites_count: Optional[int]
    followers_count: Optional[int]
    listed_count: Optional[int]
    media_count: Optional[int]
    profile_image_url: Optional[str]
    statuses_count: Optional[int]
    verified: Optional[bool]


class TwitterScraperTweet(BaseModel):
    # Available in both, scraped and api based tweets.
    user: Optional[TwitterScraperUser] = TwitterScraperUser()
    id: Optional[str]
    text: Optional[str]
    reply_count: Optional[int]
    retweet_count: Optional[int]
    like_count: Optional[int]
    view_count: Optional[int]
    quote_count: Optional[int]
    impression_count: Optional[int]
    bookmark_count: Optional[int]
    url: Optional[str]
    created_at: Optional[str]
    media: Optional[List[TwitterScraperMedia]] = []

    # Only available in scraped tweets
    is_quote_tweet: Optional[bool]
    is_retweet: Optional[bool]


class ScraperTextRole(str, Enum):
    INTRO = "intro"
    TWITTER_SUMMARY = "twitter_summary"
    SEARCH_SUMMARY = "search_summary"
    DISCORD_SUMMARY = "discord_summary"
    REDDIT_SUMMARY = "reddit_summary"
    HACKER_NEWS_SUMMARY = "hacker_news_summary"
    BITTENSOR_SUMMARY = "bittensor_summary"
    SUBNETS_SOURCE_CODE_SUMMARY = "subnets_source_code_summary"
    FINAL_SUMMARY = "summary"


class ScraperStreamingSynapse(bt.StreamingSynapse):
    prompt: str = pydantic.Field(
        ...,
        title="Prompt",
        description="The initial input or question provided by the user to guide the scraping and data collection process.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )

    # required_hash_fields: List[str] = pydantic.Field(
    #     ["messages"],
    #     title="Required Hash Fields",
    #     description="A list of required fields for the hash.",
    #     allow_mutation=False,
    # )

    # seed: int = pydantic.Field(
    #     "",
    #     title="Seed",
    #     description="Seed for text generation. This attribute is immutable and cannot be updated.",
    # )

    # model: Optional[str] = pydantic.Field(
    #     "",
    #     title="model",
    #     description="The model that which to use when calling openai for your response.",
    # )

    tools: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Tools",
        description="A list of tools specified by user to use to answer question.",
    )

    start_date: Optional[str] = pydantic.Field(
        None,
        title="Start Date",
        description="The start date for the search query.",
    )

    end_date: Optional[str] = pydantic.Field(
        None,
        title="End Date",
        description="The end date for the search query.",
    )

    date_filter_type: Optional[str] = pydantic.Field(
        None,
        title="Date filter enum",
        description="The date filter enum.",
    )

    language: Optional[str] = pydantic.Field(
        "en",
        title="Language",
        description="Language specified by user.",
    )

    region: Optional[str] = pydantic.Field(
        "us",
        title="Region",
        description="Region specified by user.",
    )

    google_date_filter: Optional[str] = pydantic.Field(
        "qdr:w",
        title="Date Filter",
        description="Date filter specified by user.",
    )

    validator_tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    validator_links: Optional[List[Dict]] = pydantic.Field(
        default_factory=list, title="Links", description="Fetched Links Data."
    )

    miner_tweets: Optional[List[Dict[str, Any]]] = pydantic.Field(
        default_factory=list,
        title="Miner Tweets",
        description="Optional JSON object containing tweets data from the miner.",
    )

    search_completion_links: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Links Content",
        description="A list of links extracted from search summary text.",
    )

    completion_links: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Links Content",
        description="A list of JSON objects representing the extracted links content from the tweets.",
    )

    search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Search Results",
        description="Optional JSON object containing search results from SERP",
    )

    google_news_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Google News Search Results",
        description="Optional JSON object containing search results from SERP Google News",
    )

    google_image_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Google Image Search Results",
        description="Optional JSON object containing image search results from SERP Google",
    )

    wikipedia_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Wikipedia Search Results",
        description="Optional JSON object containing search results from Wikipedia",
    )

    youtube_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="YouTube Search Results",
        description="Optional JSON object containing search results from YouTube",
    )

    arxiv_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Arxiv Search Results",
        description="Optional JSON object containing search results from Arxiv",
    )

    reddit_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Reddit Search Results",
        description="Optional JSON object containing search results from Reddit",
    )

    hacker_news_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Hacker News Search Results",
        description="Optional JSON object containing search results from Hacker News",
    )

    discord_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Discord Search Results",
        description="Optional JSON object containing search results from Discord",
    )

    bittensor_docs_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Bittensor Docs Search Results",
        description="Optional JSON object containing search results from Bittensor Docs",
    )

    subnets_source_code_result: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Subnets Source Code Search Results",
        description="Optional JSON object containing search results from Subnets Source Code",
    )

    text_chunks: Optional[Dict[str, List[str]]] = pydantic.Field(
        default_factory=dict,
        title="Text Chunks",
    )

    @property
    def texts(self) -> Dict[str, str]:
        """Returns a dictionary of texts, containing a role (twitter summary, search summary, reddit summary, hacker news summary, final summary) and content."""
        texts = {}

        for key in self.text_chunks:
            texts[key] = "".join(self.text_chunks[key])

        return texts

    response_order: Optional[str] = pydantic.Field(
        "",
        title="Response Order",
        description="Preffered order type of response, by default it will be SUMMARY_FIRST",
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    max_items: Optional[int] = pydantic.Field(
        None,
        title="Max Results",
        description="The maximum number of results to be returned per query",
    )

    def set_tweets(self, data: any):
        self.tweets = data

    def get_twitter_completion(self) -> Optional[str]:
        return self.texts.get(ScraperTextRole.TWITTER_SUMMARY.value, "")

    def get_search_completion(self) -> Dict[str, str]:
        """Gets the search completion text from the texts dictionary based on tools used."""

        completions = {}

        if any(
            tool in self.tools
            for tool in [
                "Google Search",
                "Google News Search",
                "Wikipedia Search",
                "Youtube Search",
                "ArXiv Search",
            ]
        ):
            search_summary = self.texts.get(ScraperTextRole.SEARCH_SUMMARY.value, "")
            completions[ScraperTextRole.SEARCH_SUMMARY.value] = search_summary

        if "Reddit Search" in self.tools:
            reddit_summary = self.texts.get(ScraperTextRole.REDDIT_SUMMARY.value, "")
            completions[ScraperTextRole.REDDIT_SUMMARY.value] = reddit_summary

        if "Hacker News Search" in self.tools:
            hacker_news_summary = self.texts.get(
                ScraperTextRole.HACKER_NEWS_SUMMARY.value, ""
            )
            completions[ScraperTextRole.HACKER_NEWS_SUMMARY.value] = hacker_news_summary

        links_per_completion = 10
        links_expected = len(completions) * links_per_completion

        completions = {key: value for key, value in completions.items() if value}

        return completions, links_expected

    def get_search_links(self) -> List[str]:
        """Extracts web links from each summary making sure to filter by domain for each tool used.
        In Reddit and Hacker News Search, the links are filtered by domains.
        In search summary part, if Google Search or Google News Search is used, the links are allowed from any domain,
        Otherwise search summary will only look for Wikipedia, ArXiv, Youtube links.
        """

        completions, _ = self.get_search_completion()
        links = []

        for key, value in completions.items():
            if key == ScraperTextRole.REDDIT_SUMMARY.value:
                links.extend(WebSearchUtils.find_links_by_domain(value, "reddit.com"))
            elif key == ScraperTextRole.HACKER_NEWS_SUMMARY.value:
                links.extend(
                    WebSearchUtils.find_links_by_domain(value, "news.ycombinator.com")
                )
            elif key == ScraperTextRole.SEARCH_SUMMARY.value:
                if any(
                    tool in self.tools
                    for tool in ["Google Search", "Google News Search"]
                ):
                    links.extend(WebSearchUtils.find_links(value))
                else:
                    if "Wikipedia Search" in self.tools:
                        links.extend(
                            WebSearchUtils.find_links_by_domain(value, "wikipedia.org")
                        )
                    if "ArXiv Search" in self.tools:
                        links.extend(
                            WebSearchUtils.find_links_by_domain(value, "arxiv.org")
                        )
                    if "Youtube Search" in self.tools:
                        links.extend(
                            WebSearchUtils.find_links_by_domain(value, "youtube.com")
                        )

        return links

    async def process_streaming_response(self, response: StreamingResponse):
        if self.completion is None:
            self.completion = ""

        buffer = ""  # Initialize an empty buffer to accumulate data across chunks

        try:
            async for chunk in response.content.iter_any():
                chunk_str = chunk.decode("utf-8", errors="ignore")

                # Attempt to parse the chunk as JSON, updating the buffer with remaining incomplete JSON data
                json_objects, buffer = extract_json_chunk(
                    chunk_str, response, self.axon.hotkey, buffer
                )
                for json_data in json_objects:
                    content_type = json_data.get("type")

                    if content_type == "text":
                        text_content = json_data.get("content", "")
                        role = json_data.get("role")

                        if role not in self.text_chunks:
                            self.text_chunks[role] = []

                        self.text_chunks[role].append(text_content)

                        yield json.dumps(
                            {"type": "text", "role": role, "content": text_content}
                        )

                    elif content_type == "completion":
                        completion = json_data.get("content", "")
                        self.completion = completion

                        yield json.dumps({"type": "completion", "content": completion})

                    elif content_type == "tweets":
                        tweets_json = json_data.get("content", "[]")
                        self.miner_tweets = tweets_json
                        yield json.dumps({"type": "tweets", "content": tweets_json})

                    elif content_type == "search":
                        search_json = json_data.get("content", "{}")
                        self.search_results = search_json
                        yield json.dumps({"type": "search", "content": search_json})

                    elif content_type == "google_search_news":
                        search_json = json_data.get("content", "{}")
                        self.google_news_search_results = search_json
                        yield json.dumps(
                            {"type": "google_search_news", "content": search_json}
                        )

                    elif content_type == "wikipedia_search":
                        search_json = json_data.get("content", "{}")
                        self.wikipedia_search_results = search_json
                        yield json.dumps(
                            {"type": "wikipedia_search", "content": search_json}
                        )

                    elif content_type == "youtube_search":
                        search_json = json_data.get("content", "{}")
                        self.youtube_search_results = search_json
                        yield json.dumps(
                            {"type": "youtube_search", "content": search_json}
                        )

                    elif content_type == "arxiv_search":
                        search_json = json_data.get("content", "{}")
                        self.arxiv_search_results = search_json
                        yield json.dumps(
                            {"type": "arxiv_search", "content": search_json}
                        )

                    elif content_type == "reddit_search":
                        search_json = json_data.get("content", "{}")
                        self.reddit_search_results = search_json
                        yield json.dumps(
                            {"type": "reddit_search", "content": search_json}
                        )

                    elif content_type == "hacker_news_search":
                        search_json = json_data.get("content", "{}")
                        self.hacker_news_search_results = search_json
                        yield json.dumps(
                            {"type": "hacker_news_search", "content": search_json}
                        )

                    elif content_type == "discord_search":
                        search_json = json_data.get("content", "{}")
                        self.discord_search_results = search_json
                        yield json.dumps(
                            {"type": "discord_search", "content": search_json}
                        )

                    elif content_type == "bittensor_docs_search":
                        search_json = json_data.get("content", "{}")
                        self.bittensor_docs_results = search_json
                        yield json.dumps(
                            {"type": "bittensor_docs_search", "content": search_json}
                        )

                    elif content_type == "subnets_source_code_search":
                        search_json = json_data.get("content", "{}")
                        self.subnets_source_code_result = search_json
                        yield json.dumps(
                            {
                                "type": "subnets_source_code_search",
                                "content": search_json,
                            }
                        )

                    elif content_type == "google_image_search":
                        search_json = json_data.get("content", "{}")
                        self.google_image_search_results = search_json
                        yield json.dumps(
                            {"type": "google_image_search", "content": search_json}
                        )
        except json.JSONDecodeError as e:
            port = response.real_url.port
            host = response.real_url.host
            hotkey = self.axon.hotkey
            bt.logging.debug(
                f"process_streaming_response: Host: {host}:{port}, hotkey: {hotkey}, ERROR: json.JSONDecodeError: {e}, "
            )
        except (TimeoutError, asyncio.exceptions.TimeoutError) as e:
            port = response.real_url.port
            host = response.real_url.host
            hotkey = self.axon.hotkey
            print(
                f"process_streaming_response TimeoutError: Host: {host}:{port}, hotkey: {hotkey}, Error: {e}"
            )
        except Exception as e:
            port = response.real_url.port
            host = response.real_url.host
            hotkey = self.axon.hotkey
            error_details = traceback.format_exc()
            bt.logging.debug(
                f"process_streaming_response: Host: {host}:{port}, hotkey: {hotkey}, ERROR: {e}, DETAILS: {error_details}, chunk: {chunk}"
            )

    def deserialize(self) -> str:
        return self.completion

    def extract_response_json(self, response: ClientResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        completion_links = TwitterUtils().find_twitter_links(self.completion)
        search_completion_links = self.get_search_links()

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "prompt": self.prompt,
            # "model": self.model,
            "completion": self.completion,
            "miner_tweets": self.miner_tweets,
            "search_results": self.search_results,
            "google_news_search_results": self.google_news_search_results,
            "wikipedia_search_results": self.wikipedia_search_results,
            "youtube_search_results": self.youtube_search_results,
            "arxiv_search_results": self.arxiv_search_results,
            "hacker_news_search_results": self.hacker_news_search_results,
            "reddit_search_results": self.reddit_search_results,
            # "prompt_analysis": self.prompt_analysis.dict(),
            "completion_links": completion_links,
            "search_completion_links": search_completion_links,
            "texts": self.texts,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "date_filter_type": self.date_filter_type,
            "tools": self.tools,
            "max_execution_time": self.max_execution_time,
            "text_chunks": self.text_chunks,
        }

    class Config:
        arbitrary_types_allowed = True


def extract_json_chunk(chunk, response, hotkey, buffer=""):
    """
    Extracts JSON objects from a chunk of data, handling cases where JSON objects are split across multiple chunks.

    :param chunk: The current chunk of data to process.
    :param response: The response object, used for logging.
    :param buffer: A buffer holding incomplete JSON data from previous chunks.
    :return: A tuple containing a list of extracted JSON objects and the updated buffer.
    """
    buffer += chunk  # Add the current chunk to the buffer
    json_objects = []

    while True:
        try:
            json_obj, end = json.JSONDecoder(strict=False).raw_decode(buffer)
            json_objects.append(json_obj)
            buffer = buffer[end:]
        except json.JSONDecodeError as e:
            if e.pos == len(buffer):
                # Reached the end of the buffer without finding a complete JSON object
                break
            elif e.msg.startswith("Unterminated string"):
                # Incomplete JSON object at the end of the chunk
                break
            else:
                # Invalid JSON data encountered
                port = response.real_url.port
                host = response.real_url.host
                bt.logging.debug(
                    f"Host: {host}:{port}; hotkey: {hotkey}; Failed to decode JSON object: {e} from {buffer}"
                )
                break

    return json_objects, buffer


class SearchSynapse(bt.Synapse):
    """A class to represent search api synapse"""

    query: str = pydantic.Field(
        "",
        title="query",
        description="The query to run tools with. Example: 'What are the recent sport events?'. Immutable.",
        allow_mutation=False,
    )

    tools: List[str] = pydantic.Field(
        default_factory=list,
        title="Tools",
        description="A list of tools specified by user to fetch data from. Immutable."
        "Available tools are: Google Search, Google Image Search, Hacker News Search, Reddit Search",
        allow_mutation=False,
    )

    uid: Optional[int] = pydantic.Field(
        None,
        title="UID",
        description="Optional miner uid to run. If not provided, a random miner will be selected. Immutable.",
        allow_mutation=False,
    )

    results: Optional[Dict[str, Any]] = pydantic.Field(
        default_factory=dict,
        title="Tool result dictionary",
        description="A dictionary of tool results where key is tool name and value is the result. Example: {'Google Search': {}, 'Google Image Search': {} }",
    )

    def deserialize(self) -> str:
        return self


class TwitterAPISynapseCall(Enum):
    GET_USER_FOLLOWERS = "GET_USER_FOLLOWERS"
    GET_USER_FOLLOWINGS = "GET_USER_FOLLOWINGS"
    GET_USER = "GET_USER"
    GET_USER_WITH_USERNAME = "GET_USER_WITH_USERNAME"
    SEARCH_TWEETS = "SEARCH_TWEETS"


class TwitterUserSynapse(bt.Synapse):
    """
    A class to represetn twitter api's user synapse
    """

    request_type: Optional[str] = pydantic.Field(
        None, title="Request type field to decide the method to call"
    )

    max_items: Optional[str] = pydantic.Field(
        None,
        title="Max Results",
        description="The maximum number of results to be returned per query",
    )

    user_id: Optional[str] = pydantic.Field(
        None,
        title="User ID",
        description="An optional string that's user of twitter's user id",
    )

    username: Optional[str] = pydantic.Field(
        None,
        title="User ID",
        description="An optional string that's user of twitter's username",
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    results: Optional[Dict[str, Any]] = pydantic.Field(
        default_factory=dict,
        title="Response dictionary",
        description="A dictionary of results returned by twitter api",
    )


class TwitterTweetSynapse(bt.Synapse):
    """A class to represent twitter api's tweet synapse"""

    prompt: Optional[str] = pydantic.Field(
        None,
        title="Search Terms",
        description="Search terms to search tweets with",
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )

    max_items: Optional[str] = pydantic.Field(
        None,
        title="Max Results",
        description="The maximum number of results to be returned per query",
    )

    min_retweets: Optional[str] = pydantic.Field(
        None,
        title="Minimum Retweets",
        description="Filter to get tweets with minimum number of retweets",
    )

    min_likes: Optional[str] = pydantic.Field(
        None,
        title="Minimum Likes",
        description="Filter to get tweets with minimum number of likes",
    )

    only_verified: Optional[bool] = pydantic.Field(
        None,
        title="Only Verified",
        description="Filter to get only verified users' tweets",
    )

    only_twitter_blue: Optional[bool] = pydantic.Field(
        None,
        title="Only Twitter Blue",
        description="Filter to get only twitter blue users' tweets",
    )

    only_video: Optional[bool] = pydantic.Field(
        None,
        title="Only Video",
        description="Filter to get only those tweets which has video embedded",
    )

    only_image: Optional[bool] = pydantic.Field(
        None,
        title="Only Image",
        description="Filter to get only those tweets which has image embedded",
    )

    only_quote: Optional[bool] = pydantic.Field(
        None,
        title="Only Quote",
        description="Filter to get only those tweets which has quote embedded",
    )

    start_date: Optional[str] = pydantic.Field(
        None,
        title="Start Date",
        description="Date range field for tweet, combine with end_date field to set a time range",
    )

    end_date: Optional[str] = pydantic.Field(
        None,
        title="End Date",
        description="Date range field for tweet, combine with start_date field to set a time range",
    )

    date_filter_type: Optional[str] = pydantic.Field(
        None,
        title="Date filter enum",
        description="The date filter enum.",
    )

    language: Optional[str] = pydantic.Field(
        "en",
        title="Language",
        description="Language specified by user.",
    )

    region: Optional[str] = pydantic.Field(
        "us",
        title="Region",
        description="Region specified by user.",
    )

    completion_links: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Links Content",
        description="A list of JSON objects representing the extracted links content from the tweets.",
    )

    miner_tweets: Optional[List[Dict[str, Any]]] = pydantic.Field(
        default_factory=list,
        title="Miner Tweets",
        description="Optional JSON object containing tweets data from the miner.",
    )

    validator_tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    results: Optional[Dict[str, Any]] = pydantic.Field(
        default_factory=dict,
        title="Response dictionary",
        description="A dictionary of results returned by twitter api",
    )

    def get_twitter_completion(self) -> Optional[str]:
        return self.texts.get(ScraperTextRole.TWITTER_SUMMARY.value, "")

    def deserialize(self) -> str:
        return self
