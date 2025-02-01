import torch
import random
import asyncio
import time
from typing import List, Optional
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
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=True,
        specified_uids=None,
        sort=None,
        user=None,
        start_date=None,
        end_date=None,
        lang=None,
        verified=None,
        blue_verified=None,
        is_quote=None,
        is_video=None,
        is_image=None,
        min_retweets=None,
        min_replies=None,
        min_likes=None,
    ):
        max_execution_time = self.max_execution_time

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
                query=task.compose_prompt(),
                sort=sort,
                user=user,
                start_date=start_date,
                end_date=end_date,
                lang=lang,
                verified=verified,
                blue_verified=blue_verified,
                is_quote=is_quote,
                is_video=is_video,
                is_image=is_image,
                min_retweets=min_retweets,
                min_replies=min_replies,
                min_likes=min_likes,
                max_execution_time=max_execution_time,
            )
            for task in tasks
        ]

        dendrites = [
            self.neuron.dendrite1,
            self.neuron.dendrite2,
            self.neuron.dendrite3,
        ]

        axon_groups = [axons[:80], axons[80:160], axons[160:]]
        synapse_groups = [synapses[:80], synapses[80:160], synapses[160:]]

        all_tasks = []  # List to collect all asyncio tasks
        timeout = max_execution_time + 5

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

    async def query_and_score_twitter_basic(self, strategy=QUERY_MINERS.RANDOM):

        try:
            if not len(self.neuron.available_uids):
                bt.logging.info(
                    "No available UIDs, skipping basic Twitter search task."
                )
                return

            # question generation
            prompts = await asyncio.gather(
                *[
                    QuestionsDataset().generate_basic_question_with_openai()
                    for _ in range(len(self.neuron.available_uids))
                ]
            )

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
            async_responses, uids, event, start_time = (
                await self.run_twitter_basic_search_and_score(
                    tasks=tasks,
                    strategy=strategy,
                    is_only_allowed_miner=False,
                    specified_uids=None,
                    start_date=None,
                    end_date=None,
                )
            )

            successful_responses = [response for response in async_responses]
            bt.logging.info(f"successful_responses is a : {successful_responses}")

            # Optionally, score or store responses for later use
            self.synthetic_history.append(
                (event, tasks, successful_responses, uids, start_time)
            )
            await self.score_random_synthetic_query()

        except Exception as e:
            bt.logging.error(f"Error in query_and_score_twitter_basic: {e}")
            raise

    async def score_random_synthetic_query(self):
        # Collect synthetic queries and score randomly
        synthetic_queries_collection_size = 2

        if len(self.synthetic_history) < synthetic_queries_collection_size:
            bt.logging.info(
                f"Skipping scoring random synthetic query as history length is {len(self.synthetic_history)}"
            )

            return

        event, tasks, final_synapses, uids, start_time = random.choice(
            self.synthetic_history
        )

        bt.logging.info(f"Scoring random synthetic query: {event}")

        await self.compute_rewards_and_penalties(
            event=event,
            tasks=tasks,
            responses=final_synapses,
            uids=uids,
            start_time=start_time,
            is_synthetic=True,
        )

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
            prompt = query["query"]
            sort = query["sort"]
            user = query["user"]
            start_date = query["start_date"]
            end_date = query["end_date"]
            lang = query["lang"]
            verified = query["verified"]
            blue_verified = query["blue_verified"]
            is_quote = query["is_quote"]
            is_video = query["is_video"]
            is_image = query["is_image"]
            min_retweets = query["min_retweets"]
            min_replies = query["min_replies"]
            min_likes = query["min_likes"]

            tasks = [
                SearchTask(
                    base_text=prompt,
                    task_name="twitter search",
                    task_type="twitter_search",
                    criteria=[],
                )
            ]

            # Run Twitter search and score
            async_responses, uids, event, start_time = (
                await self.run_twitter_basic_search_and_score(
                    tasks=tasks,
                    strategy=(
                        QUERY_MINERS.ALL if specified_uids else QUERY_MINERS.RANDOM
                    ),
                    is_only_allowed_miner=self.neuron.config.subtensor.network
                    != "finney",
                    specified_uids=specified_uids,
                    sort=sort,
                    user=user,
                    start_date=start_date,
                    end_date=end_date,
                    lang=lang,
                    verified=verified,
                    blue_verified=blue_verified,
                    is_quote=is_quote,
                    is_video=is_video,
                    is_image=is_image,
                    min_retweets=min_retweets,
                    min_replies=min_replies,
                    min_likes=min_likes,
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

            # Schedule scoring task
            asyncio.create_task(process_and_score_responses(uids))

        except Exception as e:
            bt.logging.error(f"Error in organic: {e}")
            raise e

    async def twitter_search(
        self,
        query: str,
        sort: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lang: Optional[str] = None,
        verified: Optional[bool] = None,
        blue_verified: Optional[bool] = None,
        is_quote: Optional[bool] = None,
        is_video: Optional[bool] = None,
        is_image: Optional[bool] = None,
        min_retweets: Optional[int] = None,
        min_replies: Optional[int] = None,
        min_likes: Optional[int] = None,
    ):
        """
        Perform a Twitter search using basic parameters.

        Parameters:
            query (str): The search query string, e.g., "from:user bitcoin".
            sort (str, optional): Sort by "Top" or "Latest".
            start_date (str, optional): Start date in UTC (e.g., '2025-01-01').
            end_date (str, optional): End date in UTC (e.g., '2025-01-10').
            lang (str, optional): Language filter (e.g., 'en').
            verified (bool, optional): Filter for verified accounts.
            blue_verified (bool, optional): Filter for blue verified accounts.
            quote (bool, optional): Filter for quote tweets.
            video (bool, optional): Filter for tweets with videos.
            image (bool, optional): Filter for tweets with images.
            min_retweets (int, optional): Minimum number of retweets. Defaults to 0.
            min_replies (int, optional): Minimum number of replies. Defaults to 0.
            min_likes (int, optional): Minimum number of likes. Defaults to 0.

        Returns:
            List[TwitterScraperTweet]: The list of fetched tweets.
        """
        try:
            task_name = "twitter search"

            task = SearchTask(
                base_text=query,
                task_name=task_name,
                task_type="twitter_search",
                criteria=[],
            )

            if not len(self.neuron.available_uids):
                bt.logging.info("No available UIDs.")
                raise StopAsyncIteration("No available UIDs.")

            prompt = task.compose_prompt()

            bt.logging.debug("run_task", task_name)

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
            # max_execution_time = get_max_execution_time(model)
            max_execution_time = self.max_execution_time

            # Instantiate TwitterSearchSynapse with input parameters
            synapse = TwitterSearchSynapse(
                query=prompt,
                sort=sort,
                start_date=start_date,
                end_date=end_date,
                lang=lang,
                verified=verified,
                blue_verified=blue_verified,
                is_quote=is_quote,
                is_video=is_video,
                is_image=is_image,
                min_retweets=min_retweets,
                min_replies=min_replies,
                min_likes=min_likes,
                max_execution_time=max_execution_time,
                validator_tweets=[],
                results=[],
            )

            timeout = max_execution_time + 5

            synapse: TwitterSearchSynapse = await self.neuron.dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=timeout,
                deserialize=False,
            )

            return synapse.results
        except Exception as e:
            bt.logging.error(f"Error in twitter_search: {e}")
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

