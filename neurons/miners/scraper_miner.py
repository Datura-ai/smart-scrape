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
    ScraperTextRole,
)
from template.services.twitter_api_wrapper import TwitterAPIClient
from template.db import DBClient, get_random_tweets
from template.tools.search.serp_google_search_tool import SerpGoogleSearchTool
from template.tools.twitter.twitter_summary import summarize_twitter_data
from template.tools.search.search_summary import summarize_search_data

from template.tools.tool_manager import ToolManager

# from template.tools.tool_manager import ToolManager
from template.utils import save_logs_from_miner

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=60.0)

# tool_manager = ToolManager()
# asyncio.run(tool_manager.run("What are latest AI trends?"))


class ScraperMiner:
    def __init__(self, miner: any):
        self.miner = miner

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

    async def smart_scraper(self, synapse: ScraperStreamingSynapse, send: Send):
        try:
            model = synapse.model
            prompt = synapse.messages
            seed = synapse.seed
            tools = synapse.tools
            is_intro_text = synapse.is_intro_text

            bt.logging.trace(synapse)

            bt.logging.info(
                "================================== Prompt ==================================="
            )
            bt.logging.info(prompt)
            bt.logging.info(
                "================================== Prompt ===================================="
            )

            tool_manager = ToolManager(
                prompt=prompt,
                manual_tool_names=tools,
                send=send,
                model=model,
                is_intro_text=is_intro_text,
                miner=self.miner,
            )

            await tool_manager.run()

            # TODO fix tweets data here
            # await save_logs_from_miner(
            #     self,
            #     synapse=synapse,
            #     prompt=prompt,
            #     completion=response_streamer.get_full_text(),
            #     prompt_analysis=prompt_analysis,
            #     data=tweets,
            # )

            bt.logging.info("End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in twitter scraper {e}\n{traceback.format_exc()}")
