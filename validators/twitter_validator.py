
import math
import torch
import wandb
import random
import bittensor as bt
from base_validator import BaseValidator
from template.protocol import TwitterScraper, TwitterQueryResult, StreamPrompting
from reward import (
    RewardModelType,
)
from typing import List
from utils.mock import MockRewardModel
import time
from penalty import (
    TaskValidationPenaltyModel,
    AccuracyPenaltyModel
)
from reward.open_assistant import OpenAssistantRewardModel
from reward.prompt import PromptRewardModel
from reward.dpo import DirectPreferenceRewardModel
from utils.tasks import Task, TwitterTask
# from utils import check_uid_availability, get_random_uids
from template.utils import analyze_twitter_query
from template.utils import get_random_tweet_prompts
from template.services.twilio import TwitterAPIClient

class TwitterScraperValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=60)
        self.streaming = True
        self.query_type = "text"
        self.model = "gpt-4-1106-preview"
        self.weight = 1
        self.seed = 1234

        # Init device.
        bt.logging.debug("loading", "device")
        self.device = torch.device(self.config.neuron.device)
        bt.logging.debug(str(self.device))

        self.reward_weights = torch.tensor(
            [
                self.config.reward.rlhf_weight,
                self.config.reward.prompt_based_weight,
                self.config.reward.dpo_weight,
            ],
            dtype=torch.float32,
        ).to(self.device)


        if self.reward_weights.sum() != 1:
            message = (
                f"Reward function weights do not sum to 1 (Current sum: {self.reward_weights.sum()}.)"
                f"Check your reward config file at `reward/config.py` or ensure that all your cli reward flags sum to 1."
            )
            bt.logging.error(message)
            raise Exception(message)
    
        self.reward_functions = [
            OpenAssistantRewardModel(device=self.device)
            if self.config.reward.rlhf_weight > 0
            else MockRewardModel(RewardModelType.rlhf.value),  

            PromptRewardModel(device=self.device)
            if self.config.reward.prompt_based_weight > 0
            else MockRewardModel(RewardModelType.prompt.value),

            DirectPreferenceRewardModel(device=self.device)
            if self.config.reward.dpo_weight > 0
            else MockRewardModel(RewardModelType.prompt.value),                
        ]

        self.penalty_functions = [
            TaskValidationPenaltyModel(max_penalty=0.6),
            AccuracyPenaltyModel(max_penalty=1),
        ]

        self.wandb_data = {
            "modality": "twitter_scrapper",
            "prompts": {},
            "responses": {},
            "scores": {},
            "timestamps": {},
        }

        self.twillio_api = TwitterAPIClient()
    
    async def start_query(self, available_uids, metagraph):
        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((metagraph.n)).to(self.device)
        bt.logging.debug(str(self.moving_averaged_scores))

        task_name = "augment"
        prompt = get_random_tweet_prompts(1)[0]

        # query_result: TwitterQueryResult = await analyze_twitter_query(prompt)
        query_result: TwitterQueryResult = await self.twillio_api.analyze_twitter_query(prompt=prompt)
        twitter_task = TwitterTask(base_text=prompt, task_name=task_name, task_type="twitter_scraper", criteria=[])
        twitter_task.query_result = query_result

        scores, uid_scores_dict, self.wandb_data, event = await self.run_task_and_score(
            task=twitter_task,
            available_uids=available_uids,
            metagraph=metagraph
        )
        return scores, uid_scores_dict, self.wandb_data,

    async def process_async_responses(self, async_responses):
            responses = []
            for resp in async_responses:
                full_response = ""
                synapse_object = None
                async for chunk in resp:
                    if isinstance(chunk, str):
                        bt.logging.trace(chunk)
                        full_response += chunk
                    elif isinstance(chunk, bt.Synapse):
                        synapse_object = chunk
                if synapse_object is not None:
                    synapse_object.completion = full_response
                    responses.append(synapse_object)
            return responses

    async def run_task_and_score(self, task: Task, available_uids, metagraph):
        task_name = task.task_name
        prompt = task.compose_prompt()

        bt.logging.debug("run_task", task_name)

        # Record event start time.
        event = {"name": task_name, "task_type": task.task_type}
        start_time = time.time()
        # Get the list of uids to query for this step.

        av_uids = [uid for uid in available_uids]
        uids = torch.tensor(random.sample(av_uids, len(av_uids)))
        # uids = get_random_uids(self, k=k, exclude=exclude).to(self.device)
        axons = [metagraph.axons[uid] for uid in uids] #temp todo
        synapse = TwitterScraper(messages=prompt, model=self.model, seed=self.seed)

        # Make calls to the network with the prompt.
        async_responses: List[bt.Synapse] = await self.dendrite(
            axons=axons,
            synapse=synapse,
            timeout=self.timeout,
            streaming=self.streaming,
            deserialize=False,
        )

        responses = await self.process_async_responses(async_responses)

        # Compute the rewards for the responses given the prompt.
        rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(
            self.device
        )
        for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
            reward_i_normalized, reward_event = reward_fn_i.apply(
                task.base_text, responses, task_name
            )
            rewards += weight_i * reward_i_normalized.to(self.device)
            if not self.config.neuron.disable_log_rewards:
                event = {**event, **reward_event}
            bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

        for penalty_fn_i in self.penalty_functions:
            (
                raw_penalty_i,
                adjusted_penalty_i,
                applied_penalty_i,
            ) = penalty_fn_i.apply_penalties(responses, task)
            rewards *= applied_penalty_i.to(self.device)
            if not self.config.neuron.disable_log_rewards:
                event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
                event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
                event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()
            bt.logging.trace(str(penalty_fn_i.name), applied_penalty_i.tolist())


        # Find the best completion given the rewards vector.
        completions: List[str] = [comp.completion for comp in responses]
        completion_status_message: List[str] = [
            str(comp.dendrite.status_message) for comp in responses
        ]
        completion_status_codes: List[str] = [
            str(comp.dendrite.status_code) for comp in responses
        ]

        best: str = ''
        if len(responses) != 0:
            best: str = completions[rewards.argmax(dim=0)].strip()

        # Get completion times
        completion_times: List[float] = [
            comp.dendrite.process_time if comp.dendrite.process_time != None else 0
            for comp in responses
        ]

        # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
            0, uids, rewards
        ).to(self.device)
        bt.logging.info(f"Scattered reward: {torch.mean(scattered_rewards)}")

        # Update moving_averaged_scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.moving_averaged_scores.to(self.device)
        bt.logging.info(f"Moving everaged scores: {torch.mean(self.moving_averaged_scores)}")

        # Log the step event.
        event.update(
            {
                # "block": ttl_get_block(self),
                "step_length": time.time() - start_time,
                "prompt": prompt,
                "uids": uids.tolist(),
                "completions": completions,
                "completion_times": completion_times,
                "completion_status_messages": completion_status_message,
                "completion_status_codes": completion_status_codes,
                "rewards": rewards.tolist(),
                "best": best,
            }
        )
        bt.logging.debug("Run Task event:", str(event))
        bt.logging.info(f"Best Response: {event['best']}")

        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        for uid, reward, response in zip(uids, rewards.tolist(), responses):
            uid_scores_dict[uid] = reward
            scores[uid] = reward
            self.wandb_data["scores"][uid] = reward
            self.wandb_data["responses"][uid] = response.completion
            self.wandb_data["prompts"][uid] = prompt
            
        # Return the event.
        return scores, uid_scores_dict, self.wandb_data, event
    
    async def get_and_score(self, available_uids, metagraph):

        scores, uid_scores_dict, wandb_data = await self.start_query(available_uids=available_uids, metagraph=metagraph)

        return scores, uid_scores_dict, wandb_data
    
    async def score_responses(self, responses):
        ...

    async def return_tokens(self, uid, responses):
        async for resp in responses:
            if isinstance(resp, str):
                bt.logging.trace(resp)
                yield uid, resp

    async def organic(self, metagraph, query):
        prompt = query['content']
        uid = 1
        # messages.append()
        syn = TwitterScraper(messages=prompt, model=self.model, seed=self.seed)
        bt.logging.info(f"Sending {syn.model} {self.query_type} request to uid: {uid}, timeout {self.timeout}: {syn.messages}")
        self.wandb_data["prompts"][uid] = prompt
        responses = await self.dendrite(metagraph.axons[uid], syn, deserialize=False, timeout=self.timeout, streaming=self.streaming)
        
        async for response in self.return_tokens(uid, responses):
            yield response
        # for uid, messages in query.items():
        #     prompt = messages['content']
        #     # messages.append()
        #     syn = TwitterScraper(messages=prompt, model=self.model, seed=self.seed)
        #     bt.logging.info(f"Sending {syn.model} {self.query_type} request to uid: {uid}, timeout {self.timeout}: {syn.messages}")
        #     self.wandb_data["prompts"][uid] = messages
        #     responses = await self.dendrite(metagraph.axons[uid], syn, deserialize=False, timeout=self.timeout, streaming=self.streaming)
            
        #     async for response in self.return_tokens(uid, responses):
        #         yield response

