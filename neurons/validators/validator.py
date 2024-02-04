import torch
import wandb
import asyncio
import traceback
import random
import copy
import bittensor as bt
import template.utils as utils
import os
from typing import List
from template.protocol import IsAlive
from twitter_validator import TwitterScraperValidator
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

VALIDATOR_ACCESS_KEY = os.environ.get('VALIDATOR_ACCESS_KEY')

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

    twitter_validator: "TwitterScraperValidator"
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

        self.twitter_validator = TwitterScraperValidator(neuron=self)
        bt.logging.info("initialized_validators")

        # Init the event loop.
        self.loop = asyncio.get_event_loop()
        self.step = 0
        self.steps_passed = 0
        self.exclude = []

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.config.neuron.device)
        bt.logging.debug(str(self.moving_averaged_scores))
        self.prev_block = ttl_get_block(self)

    def initialize_components(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}")
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.my_uuid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor}. Run btcli register --netuid 18 and try again.")
            exit()
    
    async def check_uid(self, axon, uid, is_only_allowed_miner = False, specified_uids=None):
        """Asynchronously check if a UID is available."""
        try:
            if specified_uids and uid not in specified_uids:
                raise Exception(f"Not allowed in Specified Uids")
            
            if self.config.neuron.only_allowed_miners and axon.coldkey not in self.config.neuron.only_allowed_miners and is_only_allowed_miner and not specified_uids:
                raise Exception(f"Not allowed")
                
            response = await self.dendrite(axon, IsAlive(), deserialize=False, timeout=4)
            if response.is_success:
                bt.logging.trace(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID
            else:
                raise Exception(f"Is not active")
        except Exception as e:
            bt.logging.trace(f"Checking UID {uid}: {e}\n{traceback.format_exc()}")
            raise e

    async def get_available_uids_is_alive(self, is_only_allowed_miner=False, specified_uids=None):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        tasks = {uid.item(): self.check_uid(self.metagraph.axons[uid.item()], uid.item(), is_only_allowed_miner, specified_uids) for uid in self.metagraph.uids}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Filter out the exceptions and keep the successful results
        available_uids = [uid for uid, result in zip(tasks.keys(), results) if not isinstance(result, Exception)]

        return available_uids

    def check_uid_availability(
        self, metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int = 4096
    ) -> bool:
        """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
        Args:
            metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
            uid (int): uid to be checked
            vpermit_tao_limit (int): Validator permit tao limit
        Returns:
            bool: True if uid is available, False otherwise
        """
        # Filter non serving axons.
        if not metagraph.axons[uid].is_serving:
            return False
        # Filter validator permit > 1024 stake.
        if metagraph.validator_permit[uid]:
            if metagraph.S[uid] > vpermit_tao_limit:
                
                return False
        # Available otherwise.
        return True

    async def get_available_uids(self, k: int = None, exclude: List[int] = None) -> torch.LongTensor:
        """Returns k available random uids from the metagraph.
        Args:
            k (int): Number of uids to return.
            exclude (List[int]): List of uids to exclude from the random sampling.
        Returns:
            uids (torch.LongTensor): Randomly sampled available uids.
        Notes:
            If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
        """
        candidate_uids = []
        avail_uids = []

        for uid in range(self.metagraph.n.item()):
            uid_is_available = self.check_uid_availability(
                self.metagraph, uid, self.config.neuron.vpermit_tao_limit
            )
            uid_is_not_excluded = exclude is None or uid not in exclude

            if uid_is_available:
                avail_uids.append(uid)
                if uid_is_not_excluded:
                    candidate_uids.append(uid)

        # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
        available_uids = candidate_uids
        return available_uids
        
    async def get_uids(self, strategy=QUERY_MINERS.RANDOM, is_only_allowed_miner=False, specified_uids=None):
        # uid_list = await self.get_available_uids()
        uid_list =  await self.get_available_uids_is_alive(is_only_allowed_miner, specified_uids) #uid_list_is_live =
        if strategy == QUERY_MINERS.RANDOM:
            uids = torch.tensor([random.choice(uid_list)]) if uid_list else torch.tensor([])
        elif strategy == QUERY_MINERS.ALL:
            uids = torch.tensor(uid_list) if uid_list else torch.tensor([])
        bt.logging.info("Run uids ---------- ", uids)
        # uid_list = list(available_uids.keys())
        return uids.to(self.config.neuron.device)

    async def update_scores(self, wandb_data):
        try:
            if self.config.wandb_on:
                wandb.log(wandb_data)
           
            iterations_per_set_weights = 2
            iterations_until_update = iterations_per_set_weights - ((self.steps_passed + 1) % iterations_per_set_weights)
            bt.logging.info(f"Updating weights in {iterations_until_update} iterations.")

            if iterations_until_update == 1:
                set_weights(self)

            self.steps_passed += 1
        except Exception as e:
            bt.logging.error(f"Error in update_scores: {e}")
            raise e

    def update_moving_averaged_scores(self, uids, rewards):
        try:
            scattered_rewards = self.moving_averaged_scores.scatter(0, uids, rewards).to(self.config.neuron.device)
            average_reward = torch.mean(scattered_rewards)
            bt.logging.info(f"Scattered reward: {average_reward:.6f}")  # Rounds to 6 decimal places for logging

            alpha = self.config.neuron.moving_average_alpha
            self.moving_averaged_scores = alpha * scattered_rewards + (1 - alpha) * self.moving_averaged_scores.to(self.config.neuron.device)
            bt.logging.info(f"Moving averaged scores: {torch.mean(self.moving_averaged_scores):.6f}")  # Rounds to 6 decimal places for logging
            return scattered_rewards
        except Exception as e:
            bt.logging.error(f"Error in update_moving_averaged_scores: {e}")
            raise e

    async def query_synapse(self, strategy=QUERY_MINERS.RANDOM):
        try:
            self.metagraph = self.subtensor.metagraph( netuid = self.config.netuid )
            await self.twitter_validator.query_and_score(strategy)
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
                    coroutines = [
                        self.query_synapse(strategy)
                        for _ in range(1)
                    ]
                    await asyncio.gather(*coroutines)
                    await asyncio.sleep(interval)  # This line introduces a five-minute delay

                self.loop.run_until_complete(run_forward())

                # Resync the network state
                if should_checkpoint(self):
                    checkpoint(self)

                # # Set the weights on chain.
                # if should_set_weights(self):
                #     set_weights(self)
                #     save_state(self)
                self.prev_block = ttl_get_block(self)
                self.step += 1
        except Exception as err:
            bt.logging.error("Error in training loop", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
    
    async def run_syn_qs(self):
        if self.config.neuron.run_random_miner_syn_qs_interval > 0:
            await asyncio.sleep(5)
            self.run(self.config.neuron.run_random_miner_syn_qs_interval, QUERY_MINERS.RANDOM)

        if self.config.neuron.run_all_miner_syn_qs_interval > 0:
            await asyncio.sleep(20)
            self.run(self.config.neuron.run_all_miner_syn_qs_interval, QUERY_MINERS.ALL)



def main():
    asyncio.run(neuron().run_syn_qs())

if __name__ == "__main__":
    main()
