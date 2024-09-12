from typing import List
import random
from datura.protocol import ScraperStreamingSynapse
from datura.dataset.date_filters import DateFilter, DateFilterType
from datetime import datetime
import pytz
import bittensor as bt


class OrganicQueryState:
    def __init__(self) -> None:
        # Tracks failed organic queries and in the next synthetic query, we will penalize the miner
        self.organic_penalties = {}

        # Tracks the all organic synapses
        self.organic_history = {}

    def save_organic_queries(
        self,
        final_synapses: List[ScraperStreamingSynapse],
        uids,
        original_rewards,
    ):
        """Save the organic queries and their rewards for future reference"""

        twitter_rewards = original_rewards[0]
        search_rewards = original_rewards[1]
        summary_rewards = original_rewards[2]
        performance_rewards = original_rewards[3]

        for (
            uid_tensor,
            synapse,
            twitter_reward,
            search_reward,
            summary_reward,
            performance_reward,
        ) in zip(
            uids,
            final_synapses,
            twitter_rewards,
            search_rewards,
            summary_rewards,
            performance_rewards,
        ):
            uid = uid_tensor.item()
            hotkey = synapse.axon.hotkey

            # axon = next(axon for axon in axons if axon.hotkey == synapse.axon.hotkey)

            is_twitter_search = "Twitter Search" in synapse.tools
            is_web_search = any(
                tool in synapse.tools
                for tool in [
                    "Google Search",
                    "Google News Search",
                    "Wikipedia Search",
                    "Youtube Search",
                    "ArXiv Search",
                    "Reddit Search",
                    "Hacker News Search",
                ]
            )

            is_failed_organic = False

            # Check if organic query failed by rewards
            if (
                (performance_reward == 0 or summary_reward == 0)
                or (is_twitter_search and twitter_reward == 0)
                or (is_web_search and search_reward == 0)
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

            if not hotkey in self.organic_history:
                self.organic_history[hotkey] = []

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
        # Collect all failed synapses
        failed_synapses = []

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

        # Choose a random synapse
        hotkey, synapse = random.choice(failed_synapses)

        start_date = datetime.strptime(
            synapse.start_date, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=pytz.utc)

        end_date = datetime.strptime(synapse.end_date, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=pytz.utc
        )

        date_filter = DateFilter(
            start_date=start_date,
            end_date=end_date,
            date_filter_type=DateFilterType(synapse.date_filter_type),
        )

        neuron = next(neuron for neuron in neurons if neuron.hotkey == hotkey)
        synapse_uid = neuron.uid

        query = {
            "content": synapse.prompt,
            "tools": synapse.tools,
            "date_filter": date_filter,
        }

        # All miners to call except the one that made the query
        specified_uids = [uid for uid in uids if uid != synapse_uid]

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
