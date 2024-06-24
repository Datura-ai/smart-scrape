import time
import threading
import itertools
import bittensor as bt


class UIDManager:
    """
    A UID manager class that takes care of taking best high value miners.
    """

    def __init__(self, wallet) -> None:
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = bt.metagraph(netuid=22)
        self.max_miners_to_use = 150
        self.validator_uid = self.metagraph.hotkeys.index(
            wallet.hotkey.ss58_address
        )
        self.axon_to_use = self.metagraph.axons[self.validator_uid]

        self.init_state()
        self.start_update_thread()

    def init_state(self):
        """
        Defines initial state of for top_uids and previous uid.
        """
        # Sort the miners by their values and take the top max_miners_to_use
        self.top_uids = self.metagraph.I.argsort(
            descending=True
        )[:self.max_miners_to_use]
        self.previous_uid = 0
        # Create an infinite cycle iterator over top_uids
        self.uid_cycle = itertools.cycle(self.top_uids)

        # Advance the iterator to the previous UID's position
        for _ in range(self.previous_uid):
            next(self.uid_cycle)

    def update_state(self):
        """
        Updates the state every 40 minutes.
        """
        while True:
            # Wait 40 minutes to re-fetch updated top-miner uids.
            time.sleep(40 * 60)
            self.top_uids = self.metagraph.I.argsort(
                descending=True
            )[:self.max_miners_to_use]
            self.previous_uid = 0
            self.uid_cycle = itertools.cycle(self.top_uids)

    def start_update_thread(self):
        """
        Starts the update state thread.
        """
        update_thread = threading.Thread(target=self.update_state)
        update_thread.daemon = True
        update_thread.start()

    def get_miner_uid(self):
        """
        Get the next miner UID from the top_uids using itertools.cycle.
        """
        self.previous_uid = next(self.uid_cycle)
        return self.previous_uid
