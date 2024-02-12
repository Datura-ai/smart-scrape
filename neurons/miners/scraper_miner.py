import re
import os
import time
import copy
import wandb
import json
import pathlib
import asyncio
import template
import argparse
import requests
import threading
import traceback
import numpy as np
import pandas as pd
import bittensor as bt

from openai import OpenAI
from functools import partial
from collections import deque
from openai import AsyncOpenAI
from starlette.types import Send
from abc import ABC, abstractmethod
from transformers import GPT2Tokenizer
from config import get_config, check_config
from typing import List, Dict, Tuple, Union, Callable, Awaitable

from template.utils import get_version
from template.protocol import (
    StreamPrompting,
    IsAlive,
    TwitterScraperStreaming,
    TwitterPromptAnalysisResult,
)
from template.services.twitter_api_wrapper import TwitterAPIClient
from template.db import DBClient, get_random_tweets
from template.tools.serp.serp_google_search_tool import SerpGoogleSearchTool
from template.tools.twitter.twitter_summary import summarize_twitter_data
from template.tools.serp.serp_summary import summarize_serp_google_search_data

# from template.tools.tool_manager import ToolManager

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=60.0)


class ScraperMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def intro_text(self, model, prompt, send, is_intro_text):
        bt.logging.trace("miner.intro_text => ", self.miner.config.miner.intro_text)
        bt.logging.trace("Synapse.is_intro_text => ", is_intro_text)
        if not self.miner.config.miner.intro_text:
            return

        if not is_intro_text:
            return

        bt.logging.trace("Run intro text")

        content = f"""
        Generate introduction for that prompt: "{prompt}",

        Something like it: "To effectively address your query, my approach involves a comprehensive analysis and integration of relevant Twitter and Google web search data. Here's how it works:

        Question or Topic Analysis: I start by thoroughly examining your question or topic to understand the core of your inquiry or the specific area you're interested in.

        Twitter Data Search: Next, I delve into Twitter, seeking out information, discussions, and insights that directly relate to your prompt.
        Google search: Next, I search Google, seeking out information, discussions, and insights that directly relate to your prompt.

        Synthesis and Response: After gathering and analyzing this data, I compile my findings and craft a detailed response, which will be presented below"

        Output: Just return only introduction text without your comment
        """
        messages = [{"role": "user", "content": content}]
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            stream=True,
            # seed=seed,
        )

        response_streamer = ResponseStreamer(send=send)
        await response_streamer.stream_response(response=response, wait_time=0.1)

        return response_streamer.get_full_text()

    async def fetch_tweets(self, prompt):
        filtered_tweets = []
        prompt_analysis = None
        if self.miner.config.miner.mock_dataset:
            # todo we can find tweets based on twitter_query
            filtered_tweets = get_random_tweets(15)
        else:
            openai_query_model = self.miner.config.miner.openai_query_model
            openai_fix_query_model = self.miner.config.miner.openai_fix_query_model
            tw_client = TwitterAPIClient(
                openai_query_model=openai_query_model,
                openai_fix_query_model=openai_fix_query_model,
            )
            filtered_tweets, prompt_analysis = (
                await tw_client.analyse_prompt_and_fetch_tweets(
                    prompt, is_recent_tweets=True
                )
            )
        return filtered_tweets, prompt_analysis

    async def finalize_summary(self, prompt, model, information):
        content = f"""
            In <UserPrompt> provided User's prompt (Question).
            In <Information>, provided highlighted key information and relevant links from Twitter and Google Search.
            
            <UserPrompt>
            {prompt}
            </UserPrompt>


            <Information>
            {information}
            </Information>
        """

        system_message = """As a summary analyst, your task is to provide users with a clear and concise summary derived from the given information and the user's query.

        Output Guidelines (Tasks):
        1. Summary: Conduct a thorough analysis of <Information> in relation to <UserPrompt> and generate a comprehensive summary.

        Operational Rules:
        1. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
        2. Seamless Integration: Avoid explicitly stating "Based on the provided <Information>" in responses. Assume user awareness of the data integration process.
        3. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content},
        ]

        return await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )

    async def smart_scraper(self, synapse: TwitterScraperStreaming, send: Send):
        try:
            model = synapse.model
            prompt = synapse.messages
            seed = synapse.seed
            is_intro_text = synapse.is_intro_text
            bt.logging.trace(synapse)

            bt.logging.info(
                "================================== Prompt ==================================="
            )
            bt.logging.info(prompt)
            bt.logging.info(
                "================================== Prompt ===================================="
            )

            # tool_manager = ToolManager()
            # res = await tool_manager.run(prompt)
            # res = await tool_manager.run_old(prompt)
            # print(res)

            intro_response, (tweets, prompt_analysis), search_results = (
                await asyncio.gather(
                    self.intro_text(
                        model="gpt-3.5-turbo",
                        prompt=prompt,
                        send=send,
                        is_intro_text=is_intro_text,
                    ),
                    self.fetch_tweets(prompt),
                    SerpGoogleSearchTool().arun(prompt),
                )
            )

            bt.logging.info(
                "================================== Prompt analysis ==================================="
            )
            bt.logging.info(prompt_analysis)
            bt.logging.info(
                "================================== Prompt analysis ===================================="
            )
            # if prompt_analysis:
            #     synapse.set_prompt_analysis(prompt_analysis)

            # if not isinstance(tweets, str):
            #     tweets_json = json.dumps(tweets)
            #     synapse.set_tweets(tweets_json)
            # else:
            #     synapse.set_tweets(tweets)

            openai_summary_model = self.miner.config.miner.openai_summary_model

            twitter_response = await summarize_twitter_data(
                prompt=prompt,
                model=openai_summary_model,
                filtered_tweets=tweets,
                prompt_analysis=prompt_analysis,
            )

            search_response = await summarize_serp_google_search_data(
                prompt=prompt,
                model=openai_summary_model,
                data=search_results,
            )

            response_streamer = ResponseStreamer(send=send)

            await response_streamer.stream_response(response=twitter_response)
            await response_streamer.stream_response(response=search_response)

            final_summary = await self.finalize_summary(
                prompt, openai_summary_model, response_streamer.get_full_text()
            )

            await response_streamer.stream_response(response=final_summary)

            bt.logging.info(
                "================================== Completion Response ==================================="
            )
            bt.logging.info(
                f"{response_streamer.get_full_text()}"
            )  # Print the full text at the end
            bt.logging.info(
                "================================== Completion Response ==================================="
            )

            # Send prompt_analysis
            if prompt_analysis:
                prompt_analysis_response_body = {
                    "type": "prompt_analysis",
                    "content": prompt_analysis.dict(),
                }
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(prompt_analysis_response_body).encode(
                            "utf-8"
                        ),
                        "more_body": True,
                    }
                )
                bt.logging.info("Prompt Analysis sent")

            # Send tweets
            tweets_amount = tweets.get("meta", {}).get("result_count", 0)
            if tweets:
                tweets_response_body = {"type": "tweets", "content": tweets}
                response_streamer.more_body = False
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(tweets_response_body).encode("utf-8"),
                        "more_body": False,
                    }
                )
                bt.logging.info(f"Tweet data sent. Number of tweets: {tweets_amount}")

            if search_results:
                search_results_response_body = {
                    "type": "search",
                    "content": search_results,
                }
                response_streamer.more_body = False
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(search_results_response_body).encode(
                            "utf-8"
                        ),
                        "more_body": False,
                    }
                )
                bt.logging.info("Search results data sent")

            if response_streamer.more_body:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"",
                        "more_body": False,
                    }
                )

            bt.logging.info("End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")


class ResponseStreamer:
    def __init__(self, send: Send) -> None:
        self.buffer = []  # Reset buffer for finalizing data responses
        self.N = 1
        self.full_text = []  # Initialize a list to store all chunks of text
        self.more_body = True
        self.send = send

    async def stream_response(self, response, wait_time=None):
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            self.buffer.append(token)
            self.full_text.append(token)  # Append the token to the full_text list

            if len(self.buffer) == self.N:
                joined_buffer = "".join(self.buffer)
                text_data_json = json.dumps({"type": "text", "content": joined_buffer})

                await self.send(
                    {
                        "type": "http.response.body",
                        "body": text_data_json.encode("utf-8"),
                        "more_body": True,
                    }
                )

                if wait_time is not None:
                    await asyncio.sleep(wait_time)

                bt.logging.trace(f"Streamed tokens: {joined_buffer}")
                self.buffer = []  # Clear the buffer for the next set of tokens

    def get_full_text(self):
        return "".join(self.full_text)
