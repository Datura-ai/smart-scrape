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
    ScraperStreamingSynapse,
    TwitterPromptAnalysisResult,
)
from template.services.twitter_api_wrapper import TwitterAPIClient
from template.db import DBClient, get_random_tweets
from template.utils import save_logs

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

        bt.logging.trace(f"Run intro text")

        content = f"""
        Generate introduction for that prompt: "{prompt}",

        Something like it: "To effectively address your query, my approach involves a comprehensive analysis and integration of relevant Twitter data. Here's how it works:

        Question or Topic Analysis: I start by thoroughly examining your question or topic to understand the core of your inquiry or the specific area you're interested in.

        Twitter Data Search: Next, I delve into Twitter, seeking out information, discussions, and insights that directly relate to your prompt.

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

        N = 1
        buffer = []
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            buffer.append(token)
            if len(buffer) == N:
                joined_buffer = "".join(buffer)
                text_response_body = {"type": "text", "content": joined_buffer}
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": True,
                    }
                )
                await asyncio.sleep(0.1)  # Wait for 100 milliseconds
                bt.logging.trace(f"Streamed tokens: {joined_buffer}")
                buffer = []

        return buffer

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
                await tw_client.analyse_prompt_and_fetch_tweets(prompt)
            )
        return filtered_tweets, prompt_analysis

    async def finalize_data(self, prompt, model, filtered_tweets, prompt_analysis):
        content = f"""
                In <UserPrompt> provided User's prompt (Question).
                In <PromptAnalysis> I anaysis that prompts and generate query for API, keywords, hashtags, user_mentions.
                In <TwitterData>, Provided Twitter API fetched data.
                
                <UserPrompt>
                {prompt}
                </UserPrompt>

                <TwitterData>
                {filtered_tweets}
                </TwitterData>

                <PromptAnalysis>
                {prompt_analysis}
                </PromptAnalysis>
            """

        system_message = f"""As a Twitter data analyst, your task is to provide users with a clear and concise summary derived from the given Twitter data and the user's query.
            
               Tasks:
                3. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
                4. You would explain how you did retrieve data based on Analysis of <UserPrompt>.

                Output Guidelines (Tasks):
                1. Summary: Conduct a thorough analysis of <TwitterData> in relation to <UserPrompt> and generate a comprehensive summary.
                2. Relevant Links: Provide a selection of Twitter links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
                Synthesize insights from both the <UserPrompt> and the <TwitterData> to formulate a well-rounded response. But don't provide any twitter link, which is not related to <UserPrompt>.
                3. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
                4. You would explain how you did retrieve data based on <PromptAnalysis>.

                Operational Rules:
                1. No <TwitterData> Scenario: If no TwitterData is provided, inform the user that current Twitter insights related to their topic are unavailable.
                3. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
                4. Seamless Integration: Avoid explicitly stating "Based on the provided <TwitterData>" in responses. Assume user awareness of the data integration process.
                5. Please separate your responses into sections for easy reading.
                6. <TwitterData>.id and <TwitterData>.username you can use generate tweet link, example: [username](https://twitter.com/<username>/statuses/<Id>)
                7. Not return text like <UserPrompt>, <PromptAnalysis>, <PromptAnalysis> to your response, make response easy to understand to any user.
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
            # seed=seed,
        )

    def prepare_tweets_data_for_finalize(self, tweets):
        data = []

        users = tweets.get("includes", {}).get("users", [])

        for tweet in tweets.get("data", []):
            author_id = tweet.get("author_id")

            author = (
                next((user for user in users if user.get("id") == author_id), None)
                or {}
            )

            data.append(
                {
                    "id": tweet.get("id"),
                    "text": tweet.get("text"),
                    "author_id": tweet.get("author_id"),
                    "created_at": tweet.get("created_at"),
                    "url": "https://twitter.com/{}/status/{}".format(
                        author.get("username"), tweet.get("id")
                    ),
                    "username": author.get("username"),
                }
            )

        return data

    async def smart_scraper(self, synapse: ScraperStreamingSynapse, send: Send):
        try:
            buffer = []
            # buffer.append('Tests 1')

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

            # buffer.append('Test 2')
            intro_response, (tweets, prompt_analysis) = await asyncio.gather(
                self.intro_text(
                    model="gpt-3.5-turbo",
                    prompt=prompt,
                    send=send,
                    is_intro_text=is_intro_text,
                ),
                self.fetch_tweets(prompt),
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
            response = await self.finalize_data(
                prompt=prompt,
                model=openai_summary_model,
                filtered_tweets=self.prepare_tweets_data_for_finalize(tweets),
                prompt_analysis=prompt_analysis,
            )

            # Reset buffer for finalizing data responses
            buffer = []
            N = 1
            full_text = []  # Initialize a list to store all chunks of text
            more_body = True
            async for chunk in response:
                token = chunk.choices[0].delta.content or ""
                buffer.append(token)
                full_text.append(token)  # Append the token to the full_text list
                if len(buffer) == N:
                    joined_buffer = "".join(buffer)
                    text_data_json = json.dumps(
                        {"type": "text", "content": joined_buffer}
                    )
                    # Stream the text
                    await send(
                        {
                            "type": "http.response.body",
                            "body": text_data_json.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.trace(f"Streamed tokens: {joined_buffer}")
                    buffer = []  # Clear the buffer for the next set of tokens

            joined_full_text = "".join(full_text)  # Join all text chunks
            bt.logging.info(
                f"================================== Completion Responsed ==================================="
            )
            bt.logging.info(f"{joined_full_text}")  # Print the full text at the end
            bt.logging.info(
                f"================================== Completion Responsed ==================================="
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
                bt.logging.info(f"Prompt Analysis sent")

            # Send tweets
            tweets_amount = tweets.get("meta", {}).get("result_count", 0)
            if tweets:
                tweets_response_body = {"type": "tweets", "content": tweets}
                more_body = False
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(tweets_response_body).encode("utf-8"),
                        "more_body": False,
                    }
                )
                bt.logging.info(f"Tweet data sent. Number of tweets: {tweets_amount}")

            if more_body:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"",
                        "more_body": False,
                    }
                )

            if self.miner.config.miner.save_logs:
                await save_logs(
                    prompt=prompt,
                    completions=[joined_full_text],
                    prompt_analyses=[prompt_analysis.dict()],
                    data=[tweets],
                    miner_uids=None,
                    scores=None,
                )

            bt.logging.info(f"End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")
