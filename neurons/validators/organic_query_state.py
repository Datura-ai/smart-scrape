from typing import List
import random
from datura.protocol import ScraperStreamingSynapse
from datura.dataset.date_filters import DateFilter, DateFilterType
from datetime import datetime
import pytz
import logging

logger = logging.getLogger(__name__)

class OrganicQueryState:
    def __init__(self) -> None:
        # Tracks failed organic queries and in the next synthetic query, we will penalize the miner
        self.organic_penalties = {}
        # Tracks the  all organic synapses
        self.organic_history = {}
        logger.info("Initialized OrganicQueryState.")

    def save_organic_queries(
        self,
        final_synapses: List[ScraperStreamingSynapse],
        uids,
        original_rewards,
    ):
        """
        Save the organic queries and their rewards for future reference.
        Apply penalties based on reward evaluations.
        """
        logger.debug("Saving organic queries and evaluating penalties.")
        
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
            logger.debug(f"Processing UID: {uid}, Hotkey: {hotkey}")

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
                logger.info(
                    f"Failed organic query detected for UID: {uid}, Hotkey: {hotkey}"
                )
                # Save penalty for the miner for the next synthetic query
                previous_penalty = self.organic_penalties.get(hotkey, 0)
                self.organic_penalties[hotkey] = previous_penalty + 1
                logger.debug(
                    f"Updated organic_penalties for Hotkey: {hotkey} to {self.organic_penalties[hotkey]}"
                )
            else:
                logger.debug(
                    f"Organic query successful for UID: {uid}, Hotkey: {hotkey}"
                )

            if hotkey not in self.organic_history:
                self.organic_history[hotkey] = []
                logger.debug(f"Initialized organic_history for Hotkey: {hotkey}")

            self.organic_history[hotkey].append((synapse, is_failed_organic))
            logger.debug(
                f"Appended synapse to organic_history for Hotkey: {hotkey}. Failure status: {is_failed_organic}"
            )

    def has_penalty(self, hotkey: str) -> bool:
        """Check if the miner has a penalty and decrement it"""
        penalties = self.organic_penalties.get(hotkey, 0)
        logger.debug(f"Checking penalties for Hotkey: {hotkey}. Current penalties: {penalties}")

        if penalties > 0:
            self.organic_penalties[hotkey] -= 1
            logger.info(
                f"Penalty applied to Hotkey: {hotkey}. Remaining penalties: {self.organic_penalties[hotkey]}"
            )
            return True

        logger.debug(f"No penalties to apply for Hotkey: {hotkey}.")
        return False

    def get_random_organic_query( self, uids, neurons):
        """Gets a random organic query from the history to score with other miners"""
        logger.debug("Retrieving a random organic query for scoring.")
        # Collect all failed synapses
        failed_synapses = []
        for hotkey, synapses in self.organic_history.items():
            failed_synapses.extend(
                [(hotkey, synapse) for synapse, is_failed in synapses if is_failed]
            )

        # If there are no failed synapses, collect all synapses
        if not failed_synapses:
            logger.info("No failed synapses found. Collecting all synapses.")
            for hotkey, synapses in self.organic_history.items():
                failed_synapses.extend([(hotkey, synapse) for synapse, _ in synapses])

        # If there are still no synapses, return None
        if not failed_synapses:
            logger.warning("No synapses available for organic query scoring.")
            return None

        # Choose a random synapse
        hotkey, synapse = random.choice(failed_synapses)
        logger.debug(f"Selected synapse from Hotkey: {hotkey}")

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
        if not neuron:
            logger.error(f"No neuron found with Hotkey: {hotkey}")
            return None
        synapse_uid = neuron.uid
        logger.debug(f"Synapse UID: {synapse_uid}")

        query = {
            "content": synapse.prompt,
            "tools": synapse.tools,
            "date_filter": date_filter,
        }

        # All miners to call except the one that made the query
        specified_uids = [uid for uid in uids if uid != synapse_uid]

        self.organic_history = {}
        logger.debug(f"Prepared query for UID: {synapse_uid}. Specified UIDs: {specified_uids}")

        return synapse, query, synapse_uid, specified_uids

    def remove_deregistered_hotkeys(self, axons):
        """Called after metagraph resync to remove any hotkeys that are no longer registered"""
        logger.debug("Removing deregistered hotkeys from OrganicQueryState.")
        hotkeys = [axon.hotkey for axon in axons]
        original_history_count = len(self.organic_history)
        original_penalties_count = len(self.organic_penalties)

        # Remove hotkeys that are no longer active
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

        removed_history = original_history_count - len(self.organic_history)
        removed_penalties = original_penalties_count - len(self.organic_penalties)

        log_data = {
            "organic_history_removed": removed_history,
            "organic_penalties_removed": removed_penalties,
        }

        logger.info(
            f"Removed deregistered hotkeys from OrganicQueryState: {log_data}"
        )
