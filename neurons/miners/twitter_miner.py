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
from template.protocol import StreamPrompting, IsAlive, TwitterScraperStreaming, TwitterPromptAnalysisResult
from template.services.twitter import TwitterAPIClient
from template.db import DBClient, get_random_tweets

OpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=60.0)


class TwitterScrapperMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def intro_text(self, model, prompt, send):
        if not self.miner.config.miner.intro_text:
            return
        
        content = f"""
        Generate introduction for that prompt: "{prompt}",

        Something like it: "To effectively address your query, my approach involves a comprehensive analysis and integration of relevant Twitter data. Here's how it works:

        Question or Topic Analysis: I start by thoroughly examining your question or topic to understand the core of your inquiry or the specific area you're interested in.

        Twitter Data Search: Next, I delve into Twitter, seeking out information, discussions, and insights that directly relate to your prompt.

        Synthesis and Response: After gathering and analyzing this data, I compile my findings and craft a detailed response, which will be presented below"

        Output: Just return only introduction text without your comment
        """
        messages = [{'role': 'user', 'content': content}]
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
                response_body = {
                    "tokens": joined_buffer,
                    "prompt_analysis": '{}',
                    "tweets": "{}"
                }
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(response_body).encode("utf-8"),
                        "more_body": True,
                    }
                )
                bt.logging.info(f"Streamed tokens: {joined_buffer}")
                buffer = []

        # Send any remaining data in the buffer
        if buffer:
            joined_buffer = "".join(buffer)
            response_body = {
                "tokens": joined_buffer,
                "prompt_analysis": '{}',
                "tweets": "{}"
            }
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(response_body).encode("utf-8"),
                    "more_body": False,
                }
            )
            bt.logging.info(f"Streamed tokens: {joined_buffer}")
        return buffer

    async def fetch_tweets(self, prompt):
        filtered_tweets = []
        prompt_analysis = None
        if self.miner.config.miner.mock_dataset:
            #todo we can find tweets based on twitter_query
            filtered_tweets = get_random_tweets(15)
        else:
            tw_client  = TwitterAPIClient()
            filtered_tweets, prompt_analysis = await tw_client.analyse_prompt_and_fetch_tweets(prompt)
        return filtered_tweets, prompt_analysis

    async def finalize_data(self, prompt, model, filtered_tweets):
            content =F"""
                User Prompt Analysis and Twitter Data Integration

                User Prompt: "{prompt}"

                Twitter Data: "{filtered_tweets}"

                Tasks:
                1. Create a Response: Analyze the user's prompt and the provided Twitter data to generate a meaningful and relevant response.
                2. Share Relevant Twitter Links: Include links to several pertinent tweets. These links will enable users to view tweet details directly.
                3. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.

                Output Guidelines:
                1. Comprehensive Analysis: Synthesize insights from both the user's prompt and the Twitter data to formulate a well-rounded response.

                Operational Rules:
                1. No Twitter Data Scenario: If no Twitter data is provided, inform the user that current Twitter insights related to their topic are unavailable.
                3. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
                4. Seamless Integration: Avoid explicitly stating "Based on the provided Twitter data" in responses. Assume user awareness of the data integration process.
                5. Please separate your responses into sections for easy reading.
            """
            messages = [{'role': 'user', 'content': content}]
            return await client.chat.completions.create(
                model= model,
                messages= messages,
                temperature= 0.1,
                stream= True,
                # seed=seed,
            )

    async def twitter_scraper(self, synapse: TwitterScraperStreaming, send: Send):
        try:
            buffer = []
            # buffer.append('Tests 1')
            
            model = synapse.model
            prompt = synapse.messages
            seed = synapse.seed
            bt.logging.info(synapse)
            bt.logging.info(f"question is {prompt} with model {model}, seed: {seed}")

            # buffer.append('Test 2')
            intro_response, (tweets, prompt_analysis) = await asyncio.gather(
                self.intro_text(model="gpt-3.5-turbo", prompt=prompt, send=send),
                self.fetch_tweets(prompt)
            )
            
            bt.logging.info("Prompt analysis ===============================================")
            bt.logging.info(prompt_analysis)
            bt.logging.info("Prompt analysis ===============================================")
            if prompt_analysis:
                synapse.set_prompt_analysis(prompt_analysis)
            synapse.set_tweets(tweets)

            response = await self.finalize_data(prompt=prompt, model=model, filtered_tweets=tweets)

            # Reset buffer for finalaze_data responses
            buffer = []
            buffer.append('\n\n')

            N = 2
            async for chunk in response:
                token = chunk.choices[0].delta.content or ""
                buffer.append(token)
                if len(buffer) == N:
                    joined_buffer = "".join(buffer)
                    # Serialize the prompt_analysis to JSON
                    prompt_analysis_json = json.dumps(synapse.prompt_analysis.dict())
                    # Prepare the response body with both the tokens and the prompt_analysis
                    response_body = {
                        "tokens": joined_buffer,
                        "prompt_analysis": prompt_analysis_json,
                        "tweets": synapse.tweets
                    }
                    # Send the response body as JSON
                    await send(
                        {
                            "type": "http.response.body",
                            "body": json.dumps(response_body).encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.info(f"Streamed tokens: {joined_buffer}")
                    # bt.logging.info(f"Prompt Analysis: {prompt_analysis_json}")
                    buffer = []

            # Send any remaining data in the buffer
            if buffer:
                joined_buffer = "".join(buffer)
                # Serialize the prompt_analysis to JSON
                prompt_analysis_json = json.dumps(synapse.prompt_analysis.dict())
                # Prepare the response body with both the tokens and the prompt_analysis
                response_body = {
                    "tokens": joined_buffer,
                    "prompt_analysis": prompt_analysis_json,
                    "tweets": synapse.tweets
                }
                # Send the response body as JSON
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(response_body).encode("utf-8"),
                        "more_body": False,
                    }
                )
                bt.logging.info(f"Streamed tokens: {joined_buffer}")
                bt.logging.info(f"Prompt Analysis: {prompt_analysis_json}")
                bt.logging.info(f"response is {response}")
        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")
