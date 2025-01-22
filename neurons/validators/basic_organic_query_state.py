from typing import List
import random
from datura.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
    WebSearchSynapse,
)
from datura.dataset.date_filters import DateFilter, DateFilterType
from datetime import datetime
import pytz
import bittensor as bt


class BasicOrganicQueryState:
    def __init__(self) -> None:
        # Tracks failed organic queries and in the next synthetic query, we will penalize the miner
        self.organic_penalties = {}

        # Tracks the all organic synapses
        self.organic_history = {}

    def save_organic_queries(
        self,
        final_synapses: List[bt.Synapse],
        uids,
        original_rewards,
    ):
        """Save the organic queries and their rewards for future reference"""

        twitter_rewards = original_rewards[0]
        performance_rewards = original_rewards[1]

        for (
            uid_tensor,
            synapse,
            twitter_reward,
            performance_reward,
        ) in zip(
            uids,
            final_synapses,
            twitter_rewards,
            performance_rewards,
        ):
            uid = uid_tensor.item()
            hotkey = synapse.axon.hotkey

            # Instead of checking synapse.tools, we now check the class
            is_twitter_search = isinstance(
                synapse,
                (
                    TwitterSearchSynapse,
                    TwitterIDSearchSynapse,
                    TwitterURLsSearchSynapse,
                ),
            )
            is_web_search = isinstance(synapse, WebSearchSynapse)

            is_failed_organic = False

            # Check if organic query failed by rewards
            if (
                performance_reward == 0
                or (is_twitter_search and twitter_reward == 0)
                or (is_web_search)
            ):

                is_failed_organic = True

            # Save penalty for the miner for the next synthetic query
            if is_failed_organic:
                bt.logging.info(
                    f"Failed organic query by miner UID: {uid}, Hotkey: {hotkey}"
                )
                self.organic_penalties[hotkey] = (
                    self.organic_penalties.get(hotkey, 0) + 1
                )

            if hotkey not in self.organic_history:
                self.organic_history[hotkey] = []

            # Append the synapse and the result of this check
            self.organic_history[hotkey].append((synapse, is_failed_organic))

    def has_penalty(self, hotkey: str) -> bool:
        """Check if the miner has a penalty and decrement it"""
        penalties = self.organic_penalties.get(hotkey, 0)

        if penalties > 0:
            self.organic_penalties[hotkey] -= 1
            return True

        return False

    def get_random_organic_query(self, uids, neurons):
        """Gets a random organic query from the history to score with other miners"""

        failed_synapses = []

        # Collect all failed synapses first
        for hotkey, synapses in self.organic_history.items():
            failed_synapses.extend(
                [(hotkey, synapse) for synapse, is_failed in synapses if is_failed]
            )

        # If there are no failed synapses, collect all synapses
        if not failed_synapses:
            for hotkey, synapses in self.organic_history.items():
                failed_synapses.extend([(hotkey, synapse) for synapse, _ in synapses])

        # If there are still no synapses, return None
        if not failed_synapses:
            return None

        # Randomly pick one
        hotkey, synapse = random.choice(failed_synapses)

        # Check synapse class type and gather content
        content = None
        start_date = None
        end_date = None

        if isinstance(synapse, TwitterSearchSynapse):
            # The 'query' field can be used as content
            content = synapse.query

            # Convert string (YYYY-MM-DD) to datetime objects, if present
            if synapse.start_date:
                start_date = datetime.strptime(synapse.start_date, "%Y-%m-%d").replace(
                    tzinfo=pytz.utc
                )

            if synapse.end_date:
                end_date = datetime.strptime(synapse.end_date, "%Y-%m-%d").replace(
                    tzinfo=pytz.utc
                )

        elif isinstance(synapse, TwitterIDSearchSynapse):
            content = synapse.id

        elif isinstance(synapse, TwitterURLsSearchSynapse):
            # 'urls' is a dict
            content = synapse.urls

        elif isinstance(synapse, WebSearchSynapse):
            content = synapse.query

        # Find the neuron's UID
        neuron = next(n for n in neurons if n.hotkey == hotkey)
        synapse_uid = neuron.uid

        # Build the final query
        query = {"content": content}
        if isinstance(synapse, TwitterSearchSynapse):
            query["start_date"] = start_date
            query["end_date"] = end_date

        # Identify other miners except the one that made the query
        specified_uids = [uid for uid in uids if uid != synapse_uid]

        # Optionally clear all history here (if desired)
        self.organic_history = {}

        return synapse, query, synapse_uid, specified_uids

    def remove_deregistered_hotkeys(self, axons):
        """Called after metagraph resync to remove any hotkeys that are no longer registered"""
        hotkeys = [axon.hotkey for axon in axons]

        original_history_count = len(self.organic_history)
        original_penalties_count = len(self.organic_penalties)

        self.organic_history = {
            hotkey: synapses
            for hotkey, synapses in self.organic_history.items()
            if hotkey in hotkeys
        }

        self.organic_penalties = {
            hotkey: penalty
            for hotkey, penalty in self.organic_penalties.items()
            if hotkey in hotkeys
        }

        log_data = {
            "organic_history": original_history_count - len(self.organic_history),
            "organic_penalties": original_penalties_count - len(self.organic_penalties),
        }

        bt.logging.info(
            f"Removed deregistered hotkeys from organic query state: {log_data}"
        )
