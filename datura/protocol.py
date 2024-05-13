import pydantic
import bittensor as bt
import typing
import json
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Awaitable, Dict, Optional, Any
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from enum import Enum

from aiohttp import ClientResponse
from datura.services.twitter_utils import TwitterUtils
from datura.services.web_search_utils import WebSearchUtils
import random


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
    id: Optional[str] = ""
    url: Optional[str] = ""
    username: Optional[str] = ""
    description: Optional[str] = ""
    created_at: Optional[str] = ""
    favourites_count: Optional[int] = 0
    followers_count: Optional[int] = 0
    listed_count: Optional[int] = 0
    media_count: Optional[int] = 0
    name: Optional[str] = ""
    profile_image_url: Optional[str] = ""
    statuses_count: Optional[int] = 0
    verified: Optional[bool] = False


class TwitterScraperTweet(BaseModel):
    user: Optional[TwitterScraperUser] = TwitterScraperUser()
    id: Optional[str] = ""
    full_text: Optional[str] = ""
    reply_count: Optional[int] = 0
    retweet_count: Optional[int] = 0
    like_count: Optional[int] = 0
    view_count: Optional[int] = 0
    quote_count: Optional[int] = 0
    url: Optional[str] = ""
    created_at: Optional[str] = ""
    is_quote_tweet: Optional[bool] = False
    is_retweet: Optional[bool] = False
    media: Optional[List[TwitterScraperMedia]] = []


class ScraperTextRole(str, Enum):
    INTRO = "intro"
    TWITTER_SUMMARY = "twitter_summary"
    SEARCH_SUMMARY = "search_summary"
    DISCORD_SUMMARY = "discord_summary"
    REDDIT_SUMMARY = "reddit_summary"
    HACKER_NEWS_SUMMARY = "hacker_news_summary"
    FINAL_SUMMARY = "summary"


class ScraperStreamingSynapse(bt.StreamingSynapse):
    messages: str = pydantic.Field(
        ...,
        title="Messages",
        description="A list of messages in the StreamPrompting scenario, each containing a role and content. Immutable.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    seed: int = pydantic.Field(
        "",
        title="Seed",
        description="Seed for text generation. This attribute is immutable and cannot be updated.",
    )

    model: Optional[str] = pydantic.Field(
        "",
        title="model",
        description="The model that which to use when calling openai for your response.",
    )

    tools: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Tools",
        description="A list of tools specified by user to use to answer question.",
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

    date_filter: Optional[str] = pydantic.Field(
        "qdr:w",
        title="Date Filter",
        description="Date filter specified by user.",
    )

    prompt_analysis: TwitterPromptAnalysisResult = pydantic.Field(
        default_factory=lambda: TwitterPromptAnalysisResult(),
        title="Prompt Analysis",
        description="Analysis of the Twitter query result.",
    )

    validator_tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    validator_links: Optional[List[Dict]] = pydantic.Field(
        default_factory=list, title="Links", description="Fetched Links Data."
    )

    miner_tweets: Optional[Dict[str, Any]] = pydantic.Field(
        default_factory=dict,
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

    is_intro_text: bool = pydantic.Field(
        False,
        title="Is Intro Text",
        description="Indicates whether the text is an introductory text.",
    )

    texts: Optional[Dict[str, str]] = pydantic.Field(
        default_factory=dict,
        title="Texts",
        description="A dictionary of texts in the StreamPrompting scenario, containing a role (intro, twitter summary, search summary, summary) and content. Immutable.",
    )

    def set_prompt_analysis(self, data: any):
        self.prompt_analysis = data

    def set_tweets(self, data: any):
        self.tweets = data

    def get_twitter_completion(self) -> Optional[str]:
        return self.texts.get(ScraperTextRole.TWITTER_SUMMARY.value, "")

    def get_search_summary_completion(self) -> Optional[str]:
        return self.texts.get(ScraperTextRole.SEARCH_SUMMARY.value, "")

    async def process_streaming_response(self, response: StreamingResponse):
        if self.completion is None:
            self.completion = ""

        buffer = ""  # Initialize an empty buffer to accumulate data across chunks

        try:
            async for chunk in response.content.iter_any():
                # Decode the chunk from bytes to a string
                chunk_str = chunk.decode("utf-8")
                # Attempt to parse the chunk as JSON, updating the buffer with remaining incomplete JSON data
                json_objects, buffer = extract_json_chunk(chunk_str, response, buffer)
                for json_data in json_objects:
                    content_type = json_data.get("type")

                    if content_type == "text":
                        text_content = json_data.get("content", "")
                        role = json_data.get("role")

                        yield json.dumps(
                            {"type": "text", "role": role, "content": text_content}
                        )
                    elif content_type == "texts":
                        texts = json_data.get("content", "")
                        self.texts = texts
                    elif content_type == "completion":
                        completion = json_data.get("content", "")
                        self.completion = completion

                        yield json.dumps({"type": "completion", "content": completion})
                    elif content_type == "prompt_analysis":
                        prompt_analysis_json = json_data.get("content", "{}")
                        prompt_analysis = TwitterPromptAnalysisResult()
                        prompt_analysis.fill(prompt_analysis_json)
                        self.set_prompt_analysis(prompt_analysis)

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

                    elif content_type == "google_image_search":
                        search_json = json_data.get("content", "{}")
                        self.google_image_search_results = search_json
                        yield json.dumps(
                            {"type": "google_image_search", "content": search_json}
                        )
        except json.JSONDecodeError as e:
            port = response.real_url.port
            host = response.real_url.host
            bt.logging.debug(
                f"process_streaming_response Host: {host}:{port} ERROR: json.JSONDecodeError: {e}, "
            )
        except TimeoutError as e:
            port = response.real_url.port
            host = response.real_url.host
            print(f"TimeoutError occurred: Host: {host}:{port}, Error: {e}")
        except Exception as e:
            port = response.real_url.port
            host = response.real_url.host
            bt.logging.debug(
                f"process_streaming_response: Host: {host}:{port} ERROR: {e}"
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
        search_completion_links = WebSearchUtils().find_links(
            self.get_search_summary_completion()
        )

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "messages": self.messages,
            "completion": self.completion,
            "miner_tweets": self.miner_tweets,
            "search_results": self.search_results,
            "google_news_search_results": self.google_news_search_results,
            "wikipedia_search_results": self.wikipedia_search_results,
            "youtube_search_results": self.youtube_search_results,
            "arxiv_search_results": self.arxiv_search_results,
            "prompt_analysis": self.prompt_analysis.dict(),
            "completion_links": completion_links,
            "search_completion_links": search_completion_links,
            "texts": self.texts,
        }

    class Config:
        arbitrary_types_allowed = True


def extract_json_chunk(chunk, response, buffer=""):
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
                bt.logging.trace(
                    f"Host: {host}:{port}; Failed to decode JSON object: {e} from {buffer}"
                )
                break

    return json_objects, buffer


class SearchSynapse(bt.Synapse):
    """A class to represent search api synapse"""

    query: str = pydantic.Field(
        "",
        title="model",
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
        return self.query


class MinerTweet(BaseModel):
    id: str
    author_id: str
    text: str
    possibly_sensitive: Optional[bool]
    edit_history_tweet_ids: List[str]
    created_at: Optional[str] = ""
    public_metrics: Dict[str, int]


class MinerTweetAuthor(BaseModel):
    id: str
    name: str
    username: str
    created_at: str
