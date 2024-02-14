import torch
import wandb
import asyncio
import concurrent
import traceback
import random
import copy
import bittensor as bt
import template.utils as utils
import os
import time
from typing import List
from template.protocol import IsAlive
from neurons.validators.scraper_validator import ScraperValidator
from config import add_args, check_config, config
from weights import init_wandb, update_weights, set_weights
from traceback import print_exception
from base_validator import AbstractNeuron
from template import QUERY_MINERS
from template.misc import ttl_get_block

from template.utils import (
    should_checkpoint,
    checkpoint,
)


class neuron(AbstractNeuron):
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    dendrite: "bt.dendrite"

    scraper_validator: "ScraperValidator"
    moving_average_scores: torch.Tensor = None
    my_uuid: int = None
    shutdown_event: asyncio.Event()

    def __init__(self):
        self.config = neuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        self.initialize_components()

        init_wandb(self)

        self.scraper_validator = ScraperValidator(neuron=self)
        bt.logging.info("initialized_validators")

        # Init the event loop.
        self.loop = asyncio.get_event_loop()
        self.step = 0
        self.steps_passed = 0
        self.exclude = []

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(
            self.config.neuron.device
        )
        bt.logging.debug(str(self.moving_averaged_scores))
        self.prev_block = ttl_get_block(self)
        self.available_uids = []
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="asyncio"
        )
        self.loop.create_task(self.update_available_uids_periodically())
        self.loop.create_task(self.update_weights_periodically())

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    def initialize_components(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}"
        )
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.my_uuid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor}. Run btcli register --netuid 18 and try again."
            )
            exit()

    async def update_weights_periodically(self):
        while True:
            try:
                if len(self.available_uids) == 0 or torch.all(
                    self.moving_averaged_scores == 0
                ):
                    await asyncio.sleep(10)
                    continue

                # await self.update_weights(self.steps_passed)
                await set_weights(self)
            except Exception as e:
                # Log the exception or handle it as needed
                bt.logging.error(
                    f"An error occurred in update_weights_periodically: {e}"
                )
                # Optionally, decide whether to continue or break the loop based on the exception
            finally:
                # Ensure the sleep is in the finally block if you want the loop to always wait,
                # even if an error occurs.
                await asyncio.sleep(self.config.neuron.update_weight_interval)

    async def update_available_uids_periodically(self):
        while True:
            start_time = time.time()
            try:
                # It's assumed run_sync_in_async is a method that correctly handles running synchronous code in async.
                # If not, ensure it's properly implemented to avoid blocking the event loop.
                self.metagraph = await self.run_sync_in_async(
                    lambda: self.subtensor.metagraph(self.config.netuid)
                )

                # Directly await the asynchronous method without intermediate assignment to self.available_uids,
                # unless it's used elsewhere.
                available_uids = await self.get_available_uids_is_alive()
                uid_list = self.shuffled(
                    list(available_uids.keys())
                )  # Ensure shuffled is properly defined to work with async.
                self.available_uids = uid_list
                bt.logging.info(
                    f"update_available_uids_periodically Number of available UIDs for periodic update: {len(uid_list)}, UIDs: {uid_list}"
                )
            except Exception as e:
                bt.logging.error(
                    f"update_available_uids_periodically Failed to update available UIDs: {e}"
                )
                # Consider whether to continue or break the loop upon certain errors.

            end_time = time.time()
            execution_time = end_time - start_time
            bt.logging.info(
                f"update_available_uids_periodically Execution time for getting available UIDs amount is: {execution_time} seconds"
            )

            await asyncio.sleep(self.config.neuron.update_available_uids_interval)

    async def check_uid(
        self,
        axon,
        uid,
        # , is_only_allowed_miner=False, specified_uids=None
    ):
        """Asynchronously check if a UID is available."""
        try:
            # if specified_uids and uid not in specified_uids:
            #     raise Exception(f"Not allowed in Specified Uids")

            # if (
            #     self.config.neuron.only_allowed_miners
            #     and axon.coldkey not in self.config.neuron.only_allowed_miners
            #     and is_only_allowed_miner
            #     and not specified_uids
            # ):
            #     raise Exception(f"Not allowed")

            response = await self.dendrite(
                axon, IsAlive(), deserialize=False, timeout=15
            )
            if response.is_success:
                bt.logging.trace(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID
            else:
                raise Exception(f"Is not active")
        except Exception as e:
            bt.logging.trace(f"Checking UID {uid}: {e}\n{traceback.format_exc()}")
            raise e

    async def get_available_uids_is_alive(
        self,
        # is_only_allowed_miner=False, specified_uids=None
    ):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        tasks = {
            uid.item(): self.check_uid(
                self.metagraph.axons[uid.item()],
                uid.item(),
                # is_only_allowed_miner,
                # specified_uids,
            )
            for uid in self.metagraph.uids
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Filter out the exceptions and keep the successful results
        available_uids = [
            uid
            for uid, result in zip(tasks.keys(), results)
            if not isinstance(result, Exception)
        ]

        return available_uids

    async def get_uids(
        self,
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=False,
        specified_uids=None,
    ):
        if len(self.available_uids) == 0:
            bt.logging.info("No available UIDs, attempting to refresh list.")
            return self.available_uids

        # Filter uid_list based on specified_uids and only_allowed_miners
        uid_list = [
            uid
            for uid in self.available_uids
            if (not specified_uids or uid in specified_uids)
            and (
                not is_only_allowed_miner
                or self.metagraph.axons[uid].coldkey
                in self.config.neuron.only_allowed_miners
            )
        ]

        if strategy == QUERY_MINERS.RANDOM:
            uids = (
                torch.tensor([random.choice(uid_list)])
                if uid_list
                else torch.tensor([])
            )
        elif strategy == QUERY_MINERS.ALL:
            uids = torch.tensor(uid_list) if uid_list else torch.tensor([])
        bt.logging.info("Run uids ---------- ", uids)
        # uid_list = list(available_uids.keys())
        return uids.to(self.config.neuron.device)

    async def update_scores(self, wandb_data):
        try:
            if self.config.wandb_on:
                wandb.log(wandb_data)

            # iterations_per_set_weights = 2
            # iterations_until_update = iterations_per_set_weights - (
            #     (self.steps_passed + 1) % iterations_per_set_weights
            # )
            # bt.logging.info(
            #     f"Updating weights in {iterations_until_update} iterations."
            # )

            # if iterations_until_update == 1:
            #     set_weights(self)

            self.steps_passed += 1
        except Exception as e:
            bt.logging.error(f"Error in update_scores: {e}")
            raise e

    def update_moving_averaged_scores(self, uids, rewards):
        try:
            scattered_rewards = self.moving_averaged_scores.scatter(
                0, uids, rewards
            ).to(self.config.neuron.device)
            average_reward = torch.mean(scattered_rewards)
            bt.logging.info(
                f"Scattered reward: {average_reward:.6f}"
            )  # Rounds to 6 decimal places for logging

            alpha = self.config.neuron.moving_average_alpha
            self.moving_averaged_scores = alpha * scattered_rewards + (
                1 - alpha
            ) * self.moving_averaged_scores.to(self.config.neuron.device)
            bt.logging.info(
                f"Moving averaged scores: {torch.mean(self.moving_averaged_scores):.6f}"
            )  # Rounds to 6 decimal places for logging
            return scattered_rewards
        except Exception as e:
            bt.logging.error(f"Error in update_moving_averaged_scores: {e}")
            raise e

    async def query_synapse(self, strategy=QUERY_MINERS.RANDOM):
        try:
            # self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
            await self.scraper_validator.query_and_score(strategy)
        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)

    def run(self, interval=300, strategy=QUERY_MINERS.RANDOM):
        bt.logging.info(f"run: interval={interval}; strategy={strategy}")
        checkpoint(self)
        try:
            while True:
                # Run multiple forwards.
                async def run_forward():
                    coroutines = [self.query_synapse(strategy) for _ in range(1)]
                    await asyncio.gather(*coroutines)
                    await asyncio.sleep(
                        interval
                    )  # This line introduces a five-minute delay

                self.loop.run_until_complete(run_forward())

                # Resync the network state
                if should_checkpoint(self):
                    checkpoint(self)

                self.prev_block = ttl_get_block(self)
                self.step += 1
        except Exception as err:
            bt.logging.error("Error in training loop", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    async def run_syn_qs(self):
        if self.config.neuron.run_random_miner_syn_qs_interval > 0:
            await asyncio.sleep(5)
            self.run(
                self.config.neuron.run_random_miner_syn_qs_interval, QUERY_MINERS.RANDOM
            )

        if self.config.neuron.run_all_miner_syn_qs_interval > 0:
            await asyncio.sleep(20)
            self.run(self.config.neuron.run_all_miner_syn_qs_interval, QUERY_MINERS.ALL)


def main():
    asyncio.run(neuron().run_syn_qs())


if __name__ == "__main__":
    main()
