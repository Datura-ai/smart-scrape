import pydantic
import bittensor as bt
import typing
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Awaitable, Dict, Optional, Any
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field


class IsAlive(bt.Synapse):
    answer: typing.Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )


class StreamPrompting(bt.StreamingSynapse):
    messages: List[Dict[str, str]] = pydantic.Field(
        ...,
        title="Messages",
        description="A list of messages in the StreamPrompting scenario, each containing a role and content. Immutable.",
        allow_mutation=False,
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

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )

    model: str = pydantic.Field(
        "",
        title="model",
        description="The model that which to use when calling openai for your response.",
    )

    async def process_streaming_response(self, response: StreamingResponse):
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8")
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> str:
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
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

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "messages": self.messages,
            "completion": self.completion,
        }


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


class TwitterScraperTweetURL(BaseModel):
    url: str = ""
    expanded_url: str = ""
    display_url: str = ""


class TwitterScraperUserMention(BaseModel):
    id_str: str = ""
    name: str = ""
    screen_name: str = ""
    profile: str = ""


class TwitterScraperMedia(BaseModel):
    media_url: str = ""
    type: str = ""


class TwitterScraperUser(BaseModel):
    id_str: str = ""
    created_at: str = ""
    default_profile_image: bool = False
    description: str = ""
    fast_followers_count: int = 0
    favourites_count: int = 0
    followers_count: int = 0
    friends_count: int = 0
    normal_followers_count: int = 0
    listed_count: int = 0
    location: str = ""
    media_count: int = 0
    has_custom_timelines: bool = False
    is_translator: bool = False
    name: str = ""
    possibly_sensitive: bool = False
    profile_banner_url: str = ""
    profile_image_url_https: str = ""
    screen_name: str = ""
    statuses_count: int = 0
    translator_type: str = ""
    verified: bool = False
    withheld_in_countries: List[str] = []


class TwitterScraperTweet(BaseModel):
    user: TwitterScraperUser = TwitterScraperUser()
    id: str = ""
    conversation_id: str = ""
    full_text: str = ""
    reply_count: int = 0
    retweet_count: int = 0
    favorite_count: int = 0
    view_count: int = 0
    quote_count: int = 0
    url: str = ""
    created_at: str = ""
    is_quote_tweet: bool = False
    is_retweet: bool = False
    is_pinned: bool = False
    is_truncated: bool = False
    hashtags: List[str] = []
    symbols: List[str] = []
    user_mentions: List[TwitterScraperUserMention] = []
    urls: List[TwitterScraperTweetURL] = []
    media: List[TwitterScraperMedia] = []


class TwitterScraperStreaming(bt.StreamingSynapse):
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

    prompt_analysis: TwitterPromptAnalysisResult = pydantic.Field(
        default_factory=lambda: TwitterPromptAnalysisResult(),
        title="Prompt Analysis",
        description="Analysis of the Twitter query result.",
    )

    tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    links_content: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Links Content",
        description="A list of JSON objects representing the extracted links content from the tweets.",
    )

    is_intro_text: bool = pydantic.Field(
        False,
        title="Is Intro Text",
        description="Indicates whether the text is an introductory text.",
    )

    def set_prompt_analysis(self, data: any):
        self.prompt_analysis = data

    def set_tweets(self, data: any):
        self.tweets = data

    async def process_streaming_response(self, response: StreamingResponse):
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8")
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> str:
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
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

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "messages": self.messages,
            "completion": self.completion,
        }

    class Config:
        arbitrary_types_allowed = True
