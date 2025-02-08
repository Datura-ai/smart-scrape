import torch
import random
import asyncio
import time
from datetime import datetime, timedelta
import pytz
from typing import Any, Dict, List
import bittensor as bt
from datura.protocol import (
    WebSearchSynapse,
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
)
from utils.mock import MockRewardModel
from datura.dataset import QuestionsDataset
from datura import QUERY_MINERS
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.reward import RewardModelType, RewardScoringType
from neurons.validators.reward.twitter_basic_search_content_relevance import (
    TwitterBasicSearchContentRelevanceModel,
)
from neurons.validators.reward.performance_reward import PerformanceRewardModel
from neurons.validators.utils.tasks import SearchTask
from neurons.validators.basic_organic_query_state import BasicOrganicQueryState
from neurons.validators.penalty.exponential_penalty import ExponentialTimePenaltyModel


class BasicScraperValidator:
    def __init__(self, neuron: AbstractNeuron):
        self.neuron = neuron
        self.timeout = 180
        self.max_execution_time = 10

        self.synthetic_history = []
        self.synthetic_and_organic_history = {}
        self.basic_organic_query_state = BasicOrganicQueryState()
        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug(
            "self.neuron.config.neuron.device = ", str(self.neuron.config.neuron.device)
        )

        # Hardcoded weights here because the advanced scraper validator implementation is based on args.
        self.twitter_content_weight = 0.70
        self.performance_weight = 0.30

        self.reward_weights = torch.tensor(
            [
                self.twitter_content_weight,
                self.performance_weight,
            ],
            dtype=torch.float32,
        ).to(self.neuron.config.neuron.device)

        if self.reward_weights.sum() != 1:
            message = (
                f"Reward function weights do not sum to 1 (Current sum: {self.reward_weights.sum()}.)"
                f"Check your reward config file at `reward/config.py` or ensure that all your cli reward flags sum to 1."
            )
            bt.logging.error(message)
            raise Exception(message)

        self.reward_functions = [
            (
                TwitterBasicSearchContentRelevanceModel(
                    device=self.neuron.config.neuron.device,
                    scoring_type=RewardScoringType.search_relevance_score_template,
                )
                if self.neuron.config.reward.twitter_content_weight > 0
                else MockRewardModel(RewardModelType.twitter_content_relevance.value)
            ),
            (
                PerformanceRewardModel(
                    device=self.neuron.config.neuron.device,
                )
                if self.neuron.config.reward.performance_weight > 0
                else MockRewardModel(RewardModelType.performance_score.value)
            ),
        ]

        self.penalty_functions = [
            ExponentialTimePenaltyModel(max_penalty=1),
        ]

    async def run_twitter_basic_search_and_score(
        self,
        tasks: List[SearchTask],
        params_list: List[Dict[str, Any]],
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=True,
        specified_uids=None,
        query_type="organic",
    ):
        event = {
            "names": [task.task_name for task in tasks],
            "task_types": [task.task_type for task in tasks],
        }

        start_time = time.time()

        uids = await self.neuron.get_uids(
            strategy=strategy,
            is_only_allowed_miner=is_only_allowed_miner,
            specified_uids=specified_uids,
        )
        axons = [self.neuron.metagraph.axons[uid] for uid in uids]

        synapses: List[TwitterSearchSynapse] = [
            TwitterSearchSynapse(
                **params,
                query=task.compose_prompt(),
                max_execution_time=self.max_execution_time,
            )
            for task, params in zip(tasks, params_list)
        ]

        dendrites = [
            self.neuron.dendrite1,
            self.neuron.dendrite2,
            self.neuron.dendrite3,
        ]

        axon_groups = [axons[:80], axons[80:160], axons[160:]]
        synapse_groups = [synapses[:80], synapses[80:160], synapses[160:]]

        all_tasks = []  # List to collect all asyncio tasks
        timeout = self.max_execution_time + 5

        for dendrite, axon_group, synapse_group in zip(
            dendrites, axon_groups, synapse_groups
        ):
            for axon, syn in zip(axon_group, synapse_group):
                # Create a task for each dendrite call
                task = dendrite.call(
                    target_axon=axon,
                    synapse=syn.copy(),
                    timeout=timeout,
                    deserialize=False,
                )
                all_tasks.append(task)

        # Await all tasks concurrently
        all_responses = await asyncio.gather(*all_tasks, return_exceptions=True)

        return all_responses, uids, event, start_time

    async def compute_rewards_and_penalties(
        self,
        event,
        tasks,
        responses,
        uids,
        start_time,
        is_synthetic=False,
    ):
        try:
            if not len(uids):
                bt.logging.warning("No UIDs provided for logging event.")
                return

            bt.logging.info("Computing rewards and penalties")

            rewards = torch.zeros(len(responses), dtype=torch.float32).to(
                self.neuron.config.neuron.device
            )

            all_rewards = []
            all_original_rewards = []
            val_score_responses_list = []

            organic_penalties = []

            bt.logging.trace(f"Received responses: {responses}")

            if is_synthetic:
                penalized_uids = []

                for uid, response in zip(uids.tolist(), responses):
                    has_penalty = self.basic_organic_query_state.has_penalty(
                        response.axon.hotkey
                    )

                    organic_penalties.append(has_penalty)

                    if has_penalty:
                        penalized_uids.append(uid)

                bt.logging.info(
                    f"Following UIDs will be penalized as they failed organic query: {penalized_uids}"
                )
            else:
                organic_penalties = [False] * len(uids)

            for weight_i, reward_fn_i in zip(
                self.reward_weights, self.reward_functions
            ):
                start_time = time.time()
                (
                    reward_i_normalized,
                    reward_event,
                    val_score_responses,
                    original_rewards,
                ) = await reward_fn_i.apply(responses, uids, organic_penalties)

                all_rewards.append(reward_i_normalized)
                all_original_rewards.append(original_rewards)
                val_score_responses_list.append(val_score_responses)

                rewards += weight_i * reward_i_normalized.to(
                    self.neuron.config.neuron.device
                )
                if not self.neuron.config.neuron.disable_log_rewards:
                    event = {**event, **reward_event}
                execution_time = time.time() - start_time
                bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())
                bt.logging.info(
                    f"Applied reward function: {reward_fn_i.name} in {execution_time / 60:.2f} minutes"
                )

            for penalty_fn_i in self.penalty_functions:
                raw_penalty_i, adjusted_penalty_i, applied_penalty_i = (
                    penalty_fn_i.apply_penalties(responses, tasks)
                )
                penalty_start_time = time.time()
                rewards *= applied_penalty_i.to(self.neuron.config.neuron.device)
                penalty_execution_time = time.time() - penalty_start_time
                if not self.neuron.config.neuron.disable_log_rewards:
                    event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
                    event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
                    event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()
                bt.logging.trace(str(penalty_fn_i.name), applied_penalty_i.tolist())
                bt.logging.info(
                    f"Applied penalty function: {penalty_fn_i.name} in {penalty_execution_time:.2f} seconds"
                )

            if is_synthetic:
                scattered_rewards = self.neuron.update_moving_averaged_scores(
                    uids, rewards
                )
                self.log_event(tasks, event, start_time, uids, rewards)

            scores = torch.zeros(len(self.neuron.metagraph.hotkeys))
            uid_scores_dict = {}
            wandb_data = {
                "modality": "twitter_scrapper",
                "prompts": {},
                "responses": {},
                "scores": {},
                "timestamps": {},
                "twitter_reward": {},
                "latency_reward": {},
            }
            bt.logging.info(
                f"======================== Reward ==========================="
            )
            # Initialize an empty list to accumulate log messages
            log_messages = []
            for uid_tensor, reward, response in zip(uids, rewards.tolist(), responses):
                uid = uid_tensor.item()

                # Accumulate log messages instead of logging them immediately
                log_messages.append(f"UID: {uid}, R: {round(reward, 3)}")

            # Log the accumulated messages in groups of three
            for i in range(0, len(log_messages), 3):
                bt.logging.info(" | ".join(log_messages[i : i + 3]))

            bt.logging.info(
                f"======================== Reward ==========================="
            )
            bt.logging.info(f"this is a all reward {all_rewards} ")

            twitter_rewards = all_rewards[0]
            latency_rewards = all_rewards[1]
            zipped_rewards = zip(
                uids,
                rewards.tolist(),
                responses,
                twitter_rewards,
                latency_rewards,
            )

            for (
                uid_tensor,
                reward,
                response,
                twitter_reward,
                latency_reward,
            ) in zipped_rewards:
                uid = uid_tensor.item()  # Convert tensor to int
                uid_scores_dict[uid] = reward
                scores[uid] = reward  # Now 'uid' is an int, which is a valid key type
                wandb_data["scores"][uid] = reward
                if hasattr(response, "query"):
                    wandb_data["prompts"][uid] = response.query
                elif hasattr(response, "id"):
                    wandb_data["prompts"][uid] = response.id
                elif hasattr(response, "urls"):
                    wandb_data["prompts"][uid] = response.urls
                wandb_data["twitter_reward"][uid] = twitter_reward
                wandb_data["latency_reward"][uid] = latency_reward

            await self.neuron.update_scores_for_basic(
                wandb_data=wandb_data,
                responses=responses,
                uids=uids,
                rewards=rewards,
                all_rewards=all_rewards,
                all_original_rewards=all_original_rewards,
                val_score_responses_list=val_score_responses_list,
                organic_penalties=organic_penalties,
                neuron=self.neuron,
            )

            return rewards, uids, val_score_responses_list, event, all_original_rewards
        except Exception as e:
            bt.logging.error(f"Error in compute_rewards_and_penalties: {e}")
            raise e

    def log_event(self, tasks, event, start_time, uids, rewards):
        event.update(
            {
                "step_length": time.time() - start_time,
                "prompts": [task.compose_prompt() for task in tasks],
                "uids": uids.tolist(),
                "rewards": rewards.tolist(),
            }
        )

        bt.logging.debug("Run Task event:", event)

    def generate_random_twitter_search_params(self) -> Dict[str, Any]:
        """
        Generate random logical parameters for Twitter search queries.
        Returns a dictionary with randomly selected parameters.
        """

        # Define which fields will be used (randomly select 1-6 fields)
        all_fields = [
            "is_quote",
            "is_video",
            "is_image",
            "min_retweets",
            "min_replies",
            "min_likes",
            "date_range",
        ]

        num_fields = random.randint(1, 3)
        selected_fields = random.sample(all_fields, num_fields)

        params: Dict[str, Any] = {}

        # Generate random date range if selected
        if "date_range" in selected_fields:
            # Generate end date (now to 1 year ago)
            end_date = datetime.now(pytz.UTC) - timedelta(days=random.randint(0, 365))

            # Randomly choose time window
            start_date = end_date - timedelta(days=random.randint(1, 7))

            params["start_date"] = start_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")
            params["end_date"] = end_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")

        # Handle media type flags (ensuring is_video and is_image aren't both True)
        if "is_video" in selected_fields and "is_image" in selected_fields:
            # If both selected, ensure they're not both True
            video_val = random.choice([True, False])

            params["is_video"] = video_val

            if video_val is False:
                params["is_image"] = random.choice([True, False])
        elif "is_video" in selected_fields:
            params["is_video"] = random.choice([True, False])
        elif "is_image" in selected_fields:
            params["is_image"] = random.choice([True, False])

        # Handle quote status
        if "is_quote" in selected_fields:
            params["is_quote"] = random.choice([True, False])

        # Handle engagement metrics with logical ranges
        if "min_likes" in selected_fields:
            params["min_likes"] = random.randint(5, 100)
        if "min_replies" in selected_fields:
            params["min_replies"] = random.randint(5, 20)
        if "min_retweets" in selected_fields:
            params["min_retweets"] = random.randint(5, 20)

        return params

    async def query_and_score_twitter_basic(self, strategy=QUERY_MINERS.RANDOM):
        try:
            if not len(self.neuron.available_uids):
                bt.logging.info(
                    "No available UIDs, skipping basic Twitter search task."
                )
                return

            dataset = QuestionsDataset()
            start_time = time.time()

            all_uids = await self.neuron.get_uids(
                strategy=QUERY_MINERS.ALL,
                is_only_allowed_miner=False,
                specified_uids=None,
            )
            called_ids = []
            bt.logging.info("checking which miners received queries in the last 4 hours")
            four_hours_in_seconds = 14400
            for _, v in self.synthetic_and_organic_history.items():
                _, _, _, ids, start_time, query_type = v
                if query_type == "organic" and start_time > time.time() - four_hours_in_seconds:
                    called_ids.extend(ids)
            specified_uids = [uid for uid in all_uids if uid not in called_ids]
            # Question generation
            prompts = await asyncio.gather(
                *[
                    dataset.generate_basic_question_with_openai()
                    for _ in range(len(self.neuron.available_uids))
                ]
            )

            params = [
                self.generate_random_twitter_search_params()
                for _ in range(len(prompts))
            ]

            # 2) Build tasks from the generated prompts
            tasks = [
                SearchTask(
                    base_text=prompt,
                    task_name="twitter search",
                    task_type="twitter_search",
                    criteria=[],
                )
                for prompt in prompts
            ]

            bt.logging.debug(
                f"[query_and_score_twitter_basic] Running with prompts: {prompts}"
            )

            # 4) Run the basic Twitter search
            responses, uids, event, start_time = (
                await self.run_twitter_basic_search_and_score(
                    tasks=tasks,
                    strategy=strategy,
                    is_only_allowed_miner=False,
                    specified_uids=specified_uids,
                    params_list=params,
                    query_type="synthetic",
                )
            )

            self.synthetic_history.append((event, tasks, responses, uids, start_time))

            for response, task, uid in zip(responses, tasks, uids):
                hotkey = response.axon.hotkey
                if hotkey not in self.synthetic_and_organic_history.keys():
                    self.synthetic_and_organic_history[hotkey] = []
                self.synthetic_and_organic_history[hotkey].append((event, task, response, uid, start_time, "synthetic"))
                bt.logging.debug(f"Saved synthetic query: {hotkey}")

            four_hours_ago = time.time() - 14400
            recent_organic_responses = []
            recent_organic_uids = []
            recent_organic_tasks = []

            hotkeys_to_pop = []
            # Collect recent organic responses
            for hotkey, history_list in self.synthetic_and_organic_history.items():
                for event, task, response, uid, timestamp, query_type in history_list:
                    if query_type == "organic" and timestamp > four_hours_ago:
                        recent_organic_responses.append(response)
                        recent_organic_uids.append(uid)
                        recent_organic_tasks.append(task)
                        hotkeys_to_pop.append(hotkey)
                        break

            organic_map = {
                uid: (response, task)
                for uid, response, task in zip(recent_organic_uids, recent_organic_responses, recent_organic_tasks)
            }

            synthetic_map = {
                uid: (response, task)
                for uid, response, task in zip(uids, responses, tasks)
            }

            all_uids = uids + recent_organic_uids

            # Combine all UIDs in correct order
            all_responses = []
            all_tasks = []
            for uid in all_uids:
                if uid in synthetic_map:
                    response, task = synthetic_map[uid]
                else:
                    response, task = organic_map[uid]
                all_responses.append(response)
                all_tasks.append(task)

            info = (event, all_tasks, all_responses, all_uids, start_time)

            for uid, response, task in zip(all_uids, all_responses, all_tasks):
                hotkey = response.axon.hotkey
                if hotkey not in self.synthetic_and_organic_history:
                    self.synthetic_and_organic_history[hotkey] = []
                query_type = "synthetic" if uid in synthetic_map else "organic"
                self.synthetic_and_organic_history[hotkey].append(
                    (event, task, response, uid, start_time, query_type)
                )
            await self.score_random_synthetic_query(info)

            bt.logging.info("Removing latest query from Synthetic and organic history")
            for hotkey in hotkeys_to_pop:
                self.synthetic_and_organic_history[hotkey].pop(0)
        except Exception as e:
            bt.logging.error(f"Error in query_and_score_twitter_basic: {e}")
            raise

    async def score_random_synthetic_query(self, info):
        # Collect synthetic queries and score randomly
        synthetic_queries_collection_size = 2

        if len(self.synthetic_history) < synthetic_queries_collection_size:
            bt.logging.info(
                f"Skipping scoring random synthetic query as history length is {len(self.synthetic_history)}"
            )

            return

        event, tasks, final_synapses, uids, start_time = info

        bt.logging.info(f"Scoring random synthetic query: {event}")

        await self.compute_rewards_and_penalties(
            event=event,
            tasks=tasks,
            responses=final_synapses,
            uids=uids,
            start_time=start_time,
            is_synthetic=True,
        )

        for _, synapse, _ in zip(tasks, final_synapses, uids):
            if synapse.axon.hotkey in self.synthetic_and_organic_history.keys():
                history = self.synthetic_and_organic_history[synapse.axon.hotkey]
                for item in history:
                    old_synapse = item[2]
                    if old_synapse.body_hash == synapse.body_hash:
                        self.synthetic_and_organic_history[synapse.axon.hotkey].remove(item)
                        break

        self.synthetic_history = []

    async def organic(
        self,
        query,
        random_synapse: TwitterSearchSynapse = None,
        random_uid=None,
        specified_uids=None,
    ):
        """Receives question from user and returns the response from the miners."""

        if not len(self.neuron.available_uids):
            bt.logging.info("No available UIDs")
            raise StopAsyncIteration("No available UIDs")

        is_interval_query = random_synapse is not None

        try:
            prompt = query.get("query", "")

            tasks = [
                SearchTask(
                    base_text=prompt,
                    task_name="twitter search",
                    task_type="twitter_search",
                    criteria=[],
                )
            ]

            async_responses, uids, event, start_time = (
                await self.run_twitter_basic_search_and_score(
                    tasks=tasks,
                    strategy=(
                        QUERY_MINERS.ALL if specified_uids else QUERY_MINERS.RANDOM
                    ),
                    is_only_allowed_miner=self.neuron.config.subtensor.network
                    != "finney",
                    specified_uids=specified_uids,
                    params_list=[
                        {key: value for key, value in query.items() if key != "query"}
                    ],
                )
            )

            final_responses = []

            # Process responses and collect successful ones
            for response in async_responses:
                if response:
                    final_responses.append(response)
                    yield response
                else:
                    bt.logging.warning(
                        f"Invalid response for UID: {response.axon.hotkey if response else 'Unknown'}"
                    )

            async def process_and_score_responses(uids):
                if is_interval_query:
                    # Add the random_synapse to final_responses and its UID to uids
                    final_responses.append(random_synapse)
                    uids = torch.cat([uids, torch.tensor([random_uid])])

                # Compute rewards and penalties
                _, _, _, _, original_rewards = await self.compute_rewards_and_penalties(
                    event=event,
                    tasks=tasks,
                    responses=final_responses,
                    uids=uids,
                    start_time=start_time,
                    is_synthetic=False,
                )

                # Save organic queries if not an interval query
                if not is_interval_query:
                    self.basic_organic_query_state.save_organic_queries(
                        final_responses, uids, original_rewards
                    )

                for response, task, uid in zip(final_responses, tasks, uids):
                    hotkey = response.axon.hotkey
                    if hotkey not in self.organic_history:
                        self.organic_history[hotkey] = []
                    self.synthetic_and_organic_history[hotkey].append((event, task, response, uid, start_time, "synthetic"))
                    bt.logging.debug(f"Saved synthetic query: {hotkey}")

            # Schedule scoring task
            asyncio.create_task(process_and_score_responses(uids))
        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e

    async def web_search(
        self,
        query: str,
        num: int = 10,
        start: int = 0,
    ):
        """
        Perform a web search using basic parameters.

        Parameters:
            query (str): The search query string, e.g., "latest news on AI".
            uid (int, optional): The unique identifier of the target axon. Defaults to None.
            num (int, optional): The maximum number of results to fetch. Defaults to 10.
            start (int, optional): The number of results to skip (used for pagination). Defaults to 0.

        Returns:
            List[WebSearchResult]: The list of fetched web results.
        """
        try:
            task_name = "web search"

            task = SearchTask(
                base_text=query,
                task_name=task_name,
                task_type="web_search",
                criteria=[],
            )

            if not len(self.neuron.available_uids):
                bt.logging.info("No available UIDs.")
                raise StopAsyncIteration("No available UIDs.")

            prompt = task.compose_prompt()

            bt.logging.debug("run_task", task_name)

            # get random uids
            uids = await self.neuron.get_uids(
                strategy=QUERY_MINERS.RANDOM,
                is_only_allowed_miner=False,
                specified_uids=None,
            )

            if uids:
                uid = uids[0]
            else:
                raise StopAsyncIteration("No available UIDs.")

            axon = self.neuron.metagraph.axons[uid]

            # Instantiate WebSearchSynapse with additional parameters
            synapse = WebSearchSynapse(
                query=prompt,
                num=num,
                start=start,
                results=[],
            )

            synapse: WebSearchSynapse = await self.neuron.dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=self.timeout,
                deserialize=False,
            )

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in search: {e}")
            raise e

    async def twitter_id_search(
        self,
        tweet_id: str,
    ):
        """
        Perform a Twitter search using a specific tweet ID, then compute rewards and save the query.
        """

        try:
            start_time = time.time()

            task_name = "twitter id search"

            task = SearchTask(
                base_text=f"Fetch tweet with ID: {tweet_id}",
                task_name=task_name,
                task_type="twitter_id_search",
                criteria=[],
            )

            if not len(self.neuron.available_uids):
                bt.logging.info("No available UIDs.")
                raise StopAsyncIteration("No available UIDs.")

            bt.logging.debug("run_task", task_name)

            uids = await self.neuron.get_uids(
                strategy=QUERY_MINERS.RANDOM,
                is_only_allowed_miner=False,
                specified_uids=None,
            )

            if not uids:
                raise StopAsyncIteration("No available UIDs.")

            uid = uids[0]

            axon = self.neuron.metagraph.axons[uid]

            synapse = TwitterIDSearchSynapse(
                id=tweet_id,
                max_execution_time=self.max_execution_time,
                validator_tweets=[],
                results=[],
            )

            timeout = self.max_execution_time + 5

            synapse: TwitterIDSearchSynapse = await self.neuron.dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=timeout,
                deserialize=False,
            )

            # 5) Build event, tasks, final_responses
            event = {
                "names": [task.task_name],
                "task_types": [task.task_type],
            }

            final_responses = [synapse]

            async def process_and_score_responses(uids_tensor):
                _, _, _, _, original_rewards = await self.compute_rewards_and_penalties(
                    event=event,
                    tasks=[task],
                    responses=final_responses,
                    uids=uids_tensor,
                    start_time=start_time,
                    is_synthetic=False,
                )

                self.basic_organic_query_state.save_organic_queries(
                    final_responses, uids_tensor, original_rewards
                )

            # Launch the scoring in the background
            uids_tensor = torch.tensor([uid], dtype=torch.int)
            asyncio.create_task(process_and_score_responses(uids_tensor))

            # 7) Return the fetched tweets
            return synapse.results

        except Exception as e:
            bt.logging.error(f"Error in ID search: {e}")
            raise e

    async def twitter_urls_search(
        self,
        urls: List[str],
    ):
        """
        Perform a Twitter search using multiple tweet URLs, then compute rewards and save the query.
        """

        try:
            start_time = time.time()

            task_name = "twitter urls search"

            if not len(self.neuron.available_uids):
                bt.logging.info("No available UIDs.")
                raise StopAsyncIteration("No available UIDs.")

            bt.logging.debug("run_task", task_name)

            # 1) Retrieve a random UID and axon
            uids = await self.neuron.get_uids(
                strategy=QUERY_MINERS.RANDOM,
                is_only_allowed_miner=False,
                specified_uids=None,
            )

            if not uids:
                raise StopAsyncIteration("No available UIDs.")

            uid = uids[0]

            axon = self.neuron.metagraph.axons[uid]

            task = SearchTask(
                base_text=f"Fetch tweets for URLs: {urls}",
                task_name=task_name,
                task_type="twitter_urls_search",
                criteria=[],
            )

            synapse = TwitterURLsSearchSynapse(
                urls=urls,
                max_execution_time=self.max_execution_time,
                validator_tweets=[],
                results=[],
            )

            timeout = self.max_execution_time + 5

            synapse: TwitterURLsSearchSynapse = await self.neuron.dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=timeout,
                deserialize=False,
            )

            # 5) Build event, tasks, final_responses
            event = {
                "names": [task.task_name],
                "task_types": [task.task_type],
            }

            final_responses = [synapse]

            async def process_and_score_responses(uids_tensor):
                _, _, _, _, original_rewards = await self.compute_rewards_and_penalties(
                    event=event,
                    tasks=[task],
                    responses=final_responses,
                    uids=uids_tensor,
                    start_time=start_time,
                    is_synthetic=False,
                )

                self.basic_organic_query_state.save_organic_queries(
                    final_responses, uids_tensor, original_rewards
                )

            uids_tensor = torch.tensor([uid], dtype=torch.int)
            asyncio.create_task(process_and_score_responses(uids_tensor))

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in URLs search: {e}")
            raise e
