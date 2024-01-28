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
                text_response_body = {
                    "type": "text",
                    "content": joined_buffer
                }
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
            #todo we can find tweets based on twitter_query
            filtered_tweets = get_random_tweets(15)
        else:
            openai_query_model = self.miner.config.miner.openai_query_model
            openai_fix_query_model = self.miner.config.miner.openai_fix_query_model
            tw_client  = TwitterAPIClient(
                openai_query_model=openai_query_model,
                openai_fix_query_model=openai_fix_query_model
            )
            filtered_tweets, prompt_analysis = await tw_client.analyse_prompt_and_fetch_tweets(prompt)
        return filtered_tweets, prompt_analysis

    async def finalize_data(self, prompt, model, filtered_tweets, prompt_analysis):
            content =F"""
                User Prompt Analysis and Twitter Data Integration
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
                
                Tasks:
                1. Create a Response: Analyze UserPrompt and the provided TwitterData to generate a meaningful and relevant response.
                2. Share Relevant Twitter Links: Include Twitter links to several pertinent tweets from provided TwitterData. These links will enable users to view tweet details directly. But only use Twitter links in your response and must return valid links.
                3. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
                4. You would explain how you did retrieve data based on Analysis of UserPrompt.

                Output Guidelines:
                1. Comprehensive Analysis: Synthesize insights from both the UserPrompt and the TwitterData to formulate a well-rounded response.

                Operational Rules:
                1. No TwitterData Scenario: If no TwitterData is provided, inform the user that current Twitter insights related to their topic are unavailable.
                3. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
                4. Seamless Integration: Avoid explicitly stating "Based on the provided TwitterData" in responses. Assume user awareness of the data integration process.
                5. Please separate your responses into sections for easy reading.
                6. TwitterData.id you can use generate tweet link, example: https://twitter.com/twitter/statuses/<Id>
                7. Not return text like UserPrompt, PromptAnalysis, PromptAnalysis to your response, make response easy to understand to any user.
            """

            system = "You are Twitter data analyst, and you have to give great summary to users based on provided Twitter data and user's prompt"
            messages = [{'role': 'system', 'content': system}, 
                        {'role': 'user', 'content': content}]
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
            is_intro_text = synapse.is_intro_text
            bt.logging.trace(synapse)
            
            bt.logging.info("================================== Prompt ===================================")
            bt.logging.info(prompt)
            bt.logging.info("================================== Prompt ====================================")

            # buffer.append('Test 2')
            intro_response, (tweets, prompt_analysis) = await asyncio.gather(
                self.intro_text(model="gpt-3.5-turbo", prompt=prompt, send=send, is_intro_text=is_intro_text),
                self.fetch_tweets(prompt)
            )
            
            bt.logging.info("================================== Prompt analysis ===================================")
            bt.logging.info(prompt_analysis)
            bt.logging.info("================================== Prompt analysis ====================================")

            openai_summary_model = self.miner.config.miner.openai_summary_model
            response = await self.finalize_data(prompt=prompt, model=openai_summary_model, filtered_tweets=tweets, prompt_analysis=prompt_analysis)

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
                    # Stream the text
                    text_response_body = {
                        "type": "text",
                        "content": joined_buffer
                    }
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.trace(f"Streamed tokens: {joined_buffer}")
                    buffer = []  # Clear the buffer for the next set of tokens

            joined_full_text = "".join(full_text)  # Join all text chunks
            bt.logging.info(f"================================== Completion Responsed ===================================") 
            bt.logging.info(f"{joined_full_text}")  # Print the full text at the end
            bt.logging.info(f"================================== Completion Responsed ===================================") 
            
            if more_body:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"",
                        "more_body": False,
                    }
                )            

            bt.logging.info(f"End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")
