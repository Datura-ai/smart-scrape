import torch
import wandb
import asyncio
import traceback
import bittensor as bt
import template.utils as utils

from template.protocol import IsAlive
from twitter_validator import TwitterScraperValidator
from config import add_args, check_config, config
from weights import init_wandb, update_weights
from traceback import print_exception

wandb_runs = {}

class neuron:
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

        self.twitter_validator = TwitterScraperValidator(
            dendrite=self.dendrite,
            config=self.config,
            subtensor=self.subtensor,
            wallet=self.wallet
        )
        bt.logging.info("initialized_validators")

        # Init the event loop.
        self.loop = asyncio.get_event_loop()
        self.step = 0

    def initialize_components(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint}")
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.my_uuid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor}. Run btcli register --netuid 18 and try again.")
            exit()

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""
        try:
            response = await self.dendrite(axon, IsAlive(), deserialize=False, timeout=4)
            if response.is_success:
                bt.logging.trace(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID
            else:
                bt.logging.trace(f"UID {uid} is not active")
                return None
        except Exception as e:
            bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
            return None

    async def get_available_uids(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        tasks = {uid.item(): self.check_uid(self.metagraph.axons[uid.item()], uid.item()) for uid in self.metagraph.uids}
        results = await asyncio.gather(*tasks.values())

        # Create a dictionary of UID to axon info for active UIDs
        available_uids = {uid: axon_info for uid, axon_info in zip(tasks.keys(), results) if axon_info is not None}
        
        return available_uids

    async def process_modality(self, selected_validator):
        available_uids = await self.get_available_uids()
        uid_list = list(available_uids.keys())
        bt.logging.info(f"starting {selected_validator.__class__.__name__} get_and_score for {uid_list}")
        scores, uid_scores_dict, wandb_data = await selected_validator.get_and_score(uid_list, self.metagraph)
        if self.config.wandb_on:
            wandb.log(wandb_data)
            bt.logging.success("wandb_log successful")
        return scores, uid_scores_dict

    async def query_synapse(self):
        try:
            steps_passed = 0
            total_scores = torch.zeros(len(self.metagraph.hotkeys))

            scores, uid_scores_dict = await self.process_modality(self.twitter_validator)
            total_scores += scores
            
            iterations_per_set_weights = 10
            iterations_until_update = iterations_per_set_weights - ((steps_passed + 1) % iterations_per_set_weights)
            bt.logging.info(f"Updating weights in {iterations_until_update} iterations.")

            if iterations_until_update == 1:
                update_weights(self, total_scores, steps_passed)

            steps_passed += 1
        except Exception as e:
            bt.logging.error(f"General exception: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(100)
    

    def run(self):
        bt.logging.info("run()")
        # load_state(self)
        # checkpoint(self)
        try:
            while True:
                if not self.wallet.hotkey.ss58_address in self.metagraph.hotkeys:
                    raise Exception(
                        f"Validator is not registered - hotkey {self.wallet.hotkey.ss58_address} not in metagraph"
                    )

                # bt.logging.info(f"step({self.step}) block({ttl_get_block( self )})")

                # Run multiple forwards.
                async def run_forward():
                    coroutines = [
                        self.query_synapse()
                        for _ in range(1)
                    ]
                    await asyncio.gather(*coroutines)
                    await asyncio.sleep(2)  # This line introduces a one-second delay

                self.loop.run_until_complete(run_forward())

                # # Resync the network state
                # if should_checkpoint(self):
                #     checkpoint(self)

                # # Set the weights on chain.
                # if should_set_weights(self):
                #     set_weights(self)
                #     save_state(self)

                # # Rollover wandb to a new run.
                # if should_reinit_wandb(self):
                #     reinit_wandb(self)

                # self.prev_block = ttl_get_block(self)
                self.step += 1
        except Exception as err:
            bt.logging.error("Error in training loop", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))




def main():
    neuron().run()

if __name__ == "__main__":
    main()
