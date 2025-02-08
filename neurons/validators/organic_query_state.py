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
        original_rewards
    ):
        """Save the organic queries and their rewards for future reference"""

        for synapse, uid, reward in zip(final_synapses, uids, original_rewards):
            query_info = {
                'model': synapse.model,
                'tools': synapse.tools,
                'date_filter_type': synapse.date_filter_type,
                'timestamp': time.time(),
                'synapse': synapse,
                'uid': uid,
                'reward': reward
            }
            self.organic_queries.append(query_info)

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

    def find_matching_queries(self, model, tools, date_filter_type):
        """
        Find organic queries that match the given model, tools, and date filter type.
        
        Args:
            model (Model): The model used for the query
            tools (List[str]): The tools used for the query
            date_filter_type (str): The date filter type enum value
        
        Returns:
            List of matching organic queries
        """
        matching_queries = []
        current_time = time.time()
        
        for query in self.organic_queries:
            if current_time - query['timestamp'] > 4 * 3600:
                continue
            
            if query['model'] != model:
                continue
            
            if set(query['tools']) != set(tools):
                continue
            
            if query['date_filter_type'] != date_filter_type:
                continue
            
            matching_queries.append(query)
        
        return matching_queries
