import threading
import time
import bittensor as bt


class UIDManager:
    """
    A UID manager class that takes care of taking best high value miners.
    """

    def __init__(self, wallet) -> None:
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = bt.metagraph(netuid=22)
        self.max_miners_to_use = 200
        self.validator_uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        self.axon_to_use = self.metagraph.axons[self.validator_uid]

        self.init_state()
        self.start_update_thread()

    def init_state(self):
        """
        Defines initial state for top_uids and previous uid.
        """
        self.top_uids = self.metagraph.I.argsort(descending=True)[:self.max_miners_to_use]
        self.uid_map = {i: uid for i, uid in enumerate(self.top_uids)}
        self.current_index = 0

    def update_state(self):
        """
        Updates the state every 40 minutes.
        """
        while True:
            # Wait 40 minutes to re-fetch updated top-miner uids.
            time.sleep(40 * 60)
            self.top_uids = self.metagraph.I.argsort(descending=True)[:self.max_miners_to_use]
            new_uid_map = {i: uid for i, uid in enumerate(self.top_uids)}

            for i in range(self.max_miners_to_use):
                self.uid_map[i] = new_uid_map.get(i, self.uid_map.get(i))

            # Ensure the current_index stays within the bounds of the new top_uids
            self.current_index %= self.max_miners_to_use

    def start_update_thread(self):
        """
        Starts the update state thread.
        """
        update_thread = threading.Thread(target=self.update_state)
        update_thread.daemon = True
        update_thread.start()

    def get_miner_uid(self):
        """
        Get the next miner UID from the top_uids using the uid_map.
        """
        miner_uid = self.uid_map[self.current_index]
        self.current_index = (self.current_index + 1) % self.max_miners_to_use
        return miner_uid
