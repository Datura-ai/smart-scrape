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
from template.services.twilio import TwitterAPIClient
from template.db import DBClient, get_random_tweets

OpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

netrc_path = pathlib.Path.home() / '.netrc'
wandb_api_key = os.getenv('WANDB_API_KEY')

print("WANDB_API_KEY is set:", bool(wandb_api_key))
print("~/.netrc exists:", netrc_path.exists())

if not wandb_api_key and not netrc_path.exists():
    raise ValueError("Please log in to wandb using `wandb login` or set the WANDB_API_KEY environment variable.")

client = AsyncOpenAI(timeout=60.0)
valid_hotkeys = []



class StreamMiner(ABC):
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        bt.logging.info("starting stream miner")
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)
        check_config(StreamMiner, self.config)
        bt.logging.info(self.config)  # TODO: duplicate print?
        self.prompt_cache: Dict[str, Tuple[str, int]] = {}
        self.request_timestamps = {}

        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"Wallet {self.wallet}")

        # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(
            f"Running miner for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint} with config:"
        )

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again. "
            )
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        # The axon handles request processing, allowing validators to send this process requests.
        self.axon = axon or bt.axon(wallet=self.wallet, port=self.config.axon.port)
        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to axon.")
        print(f"Attaching forward function to axon. {self._is_alive}")
        self.axon.attach(
            forward_fn=self._is_alive,
            blacklist_fn=self.blacklist_is_alive,
        ).attach(
            forward_fn=self._twitter_scraper,
            blacklist_fn=self.blacklist_twitter_scraper,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.request_timestamps: Dict = {}
        thread = threading.Thread(target=get_valid_hotkeys, args=(self.config,))
        # thread.start()

    @abstractmethod
    def config(self) -> "bt.Config":
        ...
    
    def _twitter_scraper(self, synapse: TwitterScraperStreaming) -> TwitterScraperStreaming:
        return self.twitter_scraper(synapse)

    def base_blacklist(self, synapse, blacklist_amt = 20000) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            if hotkey in template.WHITELISTED_KEYS:
                return False,  f"accepting {synapse_type} request from {hotkey}"

            if hotkey not in template.valid_validators:
                return True, f"Blacklisted a {synapse_type} request from a non-valid hotkey: {hotkey}"

            uid = None
            axon = None
            for _uid, _axon in enumerate(self.metagraph.axons):
                if _axon.hotkey == hotkey:
                    uid = _uid
                    axon = _axon
                    break

            if uid is None and template.ALLOW_NON_REGISTERED == False:
                return True, f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}"

            # check the stake
            tao = self.metagraph.neurons[uid].stake.tao
            # metagraph.neurons[uid].S
            if tao < blacklist_amt:
                return True, f"Blacklisted a low stake {synapse_type} request: {tao} < {blacklist_amt} from {hotkey}"

            time_window = template.MIN_REQUEST_PERIOD * 60
            current_time = time.time()

            if hotkey not in self.request_timestamps:
                self.request_timestamps[hotkey] = deque()

            # Remove timestamps outside the current time window
            while self.request_timestamps[hotkey] and current_time - self.request_timestamps[hotkey][0] > time_window:
                self.request_timestamps[hotkey].popleft()

            # Check if the number of requests exceeds the limit
            if len(self.request_timestamps[hotkey]) >= template.MAX_REQUESTS:
                return (
                    True,
                    f"Request frequency for {hotkey} exceeded: {len(self.request_timestamps[hotkey])} requests in {template.MIN_REQUEST_PERIOD} minutes. Limit is {template.MAX_REQUESTS} requests."
                )

            self.request_timestamps[hotkey].append(current_time)

            return False, f"accepting {synapse_type} request from {hotkey}"

        except Exception as e:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")

    def blacklist_is_alive( self, synapse: IsAlive ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, template.ISALIVE_BLACKLIST_STAKE)
        bt.logging.debug(blacklist[1])
        return blacklist
        
    def blacklist_twitter_scraper( self, synapse: TwitterScraperStreaming ) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse, template.TWITTER_SCRAPPER_BLACKLIST_STAKE)
        bt.logging.info(blacklist[1])
        return blacklist

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...
    
    async def _twitter_scraper(self, synapse: TwitterScraperStreaming) -> TwitterScraperStreaming:
        return self.twitter_scraper(synapse)

    def _is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("answered to be active")
        synapse.completion = "True"
        return synapse

    @abstractmethod
    def twitter_scraper(self, synapse: TwitterScraperStreaming) -> TwitterScraperStreaming:
        ...

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error( 
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
                f"Please register the hotkey using `btcli s register --netuid 18` before trying again"
            )
            exit()
        bt.logging.info(
            f"Serving axon {StreamPrompting} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info(f"Starting main loop")
        step = 0
        try:
            while not self.should_exit:
                start_epoch = time.time()

                # --- Wait until next epoch.
                current_block = self.subtensor.get_current_block()
                while (
                    current_block - self.last_epoch_block
                    < self.config.miner.blocks_per_epoch
                ):
                    # --- Wait for next bloc.
                    time.sleep(1)
                    current_block = self.subtensor.get_current_block()
                    # --- Check if we should exit.
                    if self.should_exit:
                        break

                # --- Update the metagraph with the latest network state.
                self.last_epoch_block = self.subtensor.get_current_block()

                metagraph = self.subtensor.metagraph(
                    netuid=self.config.netuid,
                    lite=True,
                    block=self.last_epoch_block,
                )
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[self.my_subnet_uid]} | "
                    f"Rank:{metagraph.R[self.my_subnet_uid]} | "
                    f"Trust:{metagraph.T[self.my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[self.my_subnet_uid] } | "
                    f"Incentive:{metagraph.I[self.my_subnet_uid]} | "
                    f"Emission:{metagraph.E[self.my_subnet_uid]}"
                )
                bt.logging.info(log)

                # --- Set weights.
                if not self.config.miner.no_set_weights:
                    pass
                step += 1

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()


class StreamingTemplateMiner(StreamMiner):
    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    def twitter_scraper(self, synapse: TwitterScraperStreaming) -> TwitterScraperStreaming:
        bt.logging.info(f"started processing for synapse {synapse}")

        async def _intro_text(model, prompt, send):
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
                        "prompt_analysis": '{}'
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
                    "prompt_analysis": '{}'
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
    
        async def _fetch_tweets(prompt):
            filtered_tweets = []
            prompt_analysis = None
            if self.config.miner.mock_dataset:
                #todo we can find tweets based on twitter_query
                filtered_tweets = get_random_tweets(15)
            else:
                tw_client  = TwitterAPIClient()
                filtered_tweets, prompt_analysis = await tw_client.analyse_prompt_and_fetch_tweets(prompt)
            return filtered_tweets, prompt_analysis
    
        async def _finalize_data(prompt, model, filtered_tweets):
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
                    2. Inclusion of Tweet Links: Incorporate 1-4 links of the most relevant tweets in your response to provide direct access and context. Provide as Bullet list
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

        async def _twitter_scraper(synapse: TwitterScraperStreaming, send: Send):
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
                    _intro_text(model="gpt-3.5-turbo", prompt=prompt, send=send),
                    _fetch_tweets(prompt)
                )
                
                bt.logging.info("Prompt analysis ===============================================")
                bt.logging.info(prompt_analysis)
                bt.logging.info("Prompt analysis ===============================================")
                if prompt_analysis:
                    synapse.set_prompt_analysis(prompt_analysis)

                response = await _finalize_data(prompt=prompt, model=model, filtered_tweets=tweets)

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
                            "prompt_analysis": prompt_analysis_json
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
                        "prompt_analysis": prompt_analysis_json
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

        token_streamer = partial(_twitter_scraper, synapse)
        return synapse.create_streaming_response(token_streamer)

def get_valid_hotkeys(config):
    global valid_hotkeys
    api = wandb.Api()
    subtensor = bt.subtensor(config=config)
    while True:
        metagraph = subtensor.metagraph(18)
        try:
            runs = api.runs(f"{template.ENTITY}/{template.PROJECT_NAME}")
            latest_version = get_version()
            for run in runs:
                if run.state == "running":
                    try:
                        # Extract hotkey and signature from the run's configuration
                        hotkey = run.config['hotkey']
                        signature = run.config['signature']
                        version = run.config['version']
                        bt.logging.debug(f"found running run of hotkey {hotkey}, {version} ")

                        if latest_version == None:
                            bt.logging.error(f'Github API call failed!')
                            continue
             
                        if version != latest_version and latest_version != None:
                            bt.logging.debug(f'Version Mismatch: Run version {version} does not match GitHub version {latest_version}')
                            continue

                        # Check if the hotkey is registered in the metagraph
                        if hotkey not in metagraph.hotkeys:
                            bt.logging.debug(f'Invalid running run: The hotkey: {hotkey} is not in the metagraph.')
                            continue

                        # Verify the signature using the hotkey
                        if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)):
                            bt.logging.debug(f'Failed Signature: The signature: {signature} is not valid')
                            continue
                            
                        if hotkey not in valid_hotkeys:
                            valid_hotkeys.append(hotkey)
                    except Exception as e:
                        bt.logging.debug(f"exception in get_valid_hotkeys: {traceback.format_exc()}")

            bt.logging.info(f"total valid hotkeys list = {valid_hotkeys}")
            time.sleep(180)

        except json.JSONDecodeError as e:
            bt.logging.debug(f"JSON decoding error: {e} {run.id}")


if __name__ == "__main__":
    with StreamingTemplateMiner():
        while True:
            time.sleep(1)
