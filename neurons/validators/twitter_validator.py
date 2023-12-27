
import math
import torch
import wandb
import random
import json
import bittensor as bt
from base_validator import BaseValidator
from template.protocol import TwitterScraperStreaming, TwitterPromptAnalysisResult
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
from utils.tasks import TwitterTask
from template.utils import get_random_tweet_prompts
from template.services.twilio import TwitterAPIClient
import asyncio

class TwitterScraperValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet, update_score, get_available_uids):
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
        self.metagraph = None
        self.neuron = None
        self.update_score = update_score
        self.get_available_uids = get_available_uids

    def set_neuron(self, metagraph, update_scores):
        self.metagraph = metagraph 
        self.update_scores = update_scores
    
    async def get_uids(self):
        available_uids = await self.get_available_uids()
        uid_list = list(available_uids.keys())
        uids = torch.tensor([random.choice(uid_list)]) if uid_list else torch.tensor([])
        return uids


    async def process_async_responses(self, async_responses):
        responses = []
        for resp in async_responses:
            full_response = ""
            synapse_object : TwitterScraperStreaming = None
            prompt_analysis = None
            async for chunk in resp:
                if isinstance(chunk, str):
                    # Parse the JSON chunk to extract tokens and prompt_analysis
                    try:
                        chunk_data = json.loads(chunk)
                        full_response += chunk_data.get("tokens", "")
                        if "prompt_analysis" in chunk_data:
                            prompt_analysis_json = chunk_data["prompt_analysis"]
                            # Assuming prompt_analysis_json is a JSON string, parse it to a Python dict
                            prompt_analysis = json.loads(prompt_analysis_json)
                    except json.JSONDecodeError:
                        bt.logging.trace(f"Failed to decode JSON chunk: {chunk}")
                elif isinstance(chunk, bt.Synapse):
                    synapse_object = chunk
            if synapse_object is not None:
                synapse_object.completion = full_response
                # Attach the prompt_analysis to the synapse_object if needed
                if prompt_analysis is not None:
                    synapse_object.set_prompt_analysis(prompt_analysis)
                responses.append(synapse_object)
        return responses

    async def run_task_and_score(self, task: TwitterTask, is_scoring_background : False):
        task_name = task.task_name
        prompt = task.compose_prompt()

        bt.logging.debug("run_task", task_name)

        # Record event start time.
        event = {"name": task_name, "task_type": task.task_type}
        start_time = time.time()
        
        # Get random id on that step
        uids = self.get_uids()
        axons = [self.metagraph.axons[uid] for uid in uids]
        synapse = TwitterScraperStreaming(messages=prompt, model=self.model, seed=self.seed)

        # Make calls to the network with the prompt.
        async_responses: List[bt.Synapse] = await self.dendrite.forward(
            axons=axons,
            synapse=synapse,
            timeout=self.timeout,
            streaming=self.streaming,
            deserialize=False,
        )

        responses = await self.process_async_responses(async_responses)
    
        if responses:
            task.prompt_analysis = responses[0].prompt_analysis

        if is_scoring_background:
            # Run compute_rewards_and_penalties as a background task
            asyncio.create_task(self.compute_rewards_and_penalties(event, task, responses, uids, start_time))
            return responses
        else:
            # Wait for compute_rewards_and_penalties to complete
            await self.compute_rewards_and_penalties(event, task, responses, uids, start_time)
            return responses

    
    async def compute_rewards_and_penalties(self, event, prompt, task, responses, uids, start_time):
        if responses:
            task.prompt_analysis = responses[0].prompt_analysis

        rewards = torch.zeros(len(responses), dtype=torch.float32).to(self.device)
        for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
            reward_i_normalized, reward_event = reward_fn_i.apply(task.base_text, responses, task.task_name)
            rewards += weight_i * reward_i_normalized.to(self.device)
            if not self.config.neuron.disable_log_rewards:
                event = {**event, **reward_event}
            bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

        for penalty_fn_i in self.penalty_functions:
            raw_penalty_i, adjusted_penalty_i, applied_penalty_i = penalty_fn_i.apply_penalties(responses, task)
            rewards *= applied_penalty_i.to(self.device)
            if not self.config.neuron.disable_log_rewards:
                event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
                event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
                event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()
            bt.logging.trace(str(penalty_fn_i.name), applied_penalty_i.tolist())

        scattered_rewards = self.update_moving_averaged_scores(uids, rewards)
        self.log_event(task, event, start_time, uids, rewards, prompt=task.compose_prompt())

        scores = torch.zeros(len(self.metagraph.hotkeys))
        uid_scores_dict = {}
        for uid, reward, response in zip(uids, rewards.tolist(), responses):
            uid_scores_dict[uid] = reward
            scores[uid] = reward
            self.wandb_data["scores"][uid] = reward
            self.wandb_data["responses"][uid] = response.completion
            self.wandb_data["prompts"][uid] = prompt
        
        self.update_scores(scores, self.wandb_data)

        return rewards, scattered_rewards

    def update_moving_averaged_scores(self, uids, rewards):
        scattered_rewards = self.moving_averaged_scores.scatter(0, uids, rewards).to(self.device)
        bt.logging.info(f"Scattered reward: {torch.mean(scattered_rewards)}")

        alpha = self.config.neuron.moving_average_alpha
        self.moving_averaged_scores = alpha * scattered_rewards + (1 - alpha) * self.moving_averaged_scores.to(self.device)
        bt.logging.info(f"Moving averaged scores: {torch.mean(self.moving_averaged_scores)}")

        return scattered_rewards

    def log_event(self, task, event, start_time, uids, rewards, prompt):
        event.update({
            "step_length": time.time() - start_time,
            "prompt": prompt,
            "uids": uids.tolist(),
            "rewards": rewards.tolist(),
        })
        bt.logging.debug("Run Task event:", str(event))
    
    async def query_and_score(self):
        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.device)
        bt.logging.debug(str(self.moving_averaged_scores))

        prompt = get_random_tweet_prompts(1)[0]

        task_name = "augment"
        twitter_task = TwitterTask(base_text=prompt, task_name=task_name, task_type="twitter_scraper", criteria=[])

        return await self.run_task_and_score(
            task=twitter_task,
            is_scoring_background=False
        )

    async def organic(self, query):
        prompt = query['content']
        uid = self.get_uids()

        task_name = "augment"
        twitter_task = TwitterTask(base_text=prompt, task_name=task_name, task_type="twitter_scraper", criteria=[])

        responses = await self.run_task_and_score(
            task=twitter_task,
            is_scoring_background=True
        )

        async for response in self.process_async_responses(uid, responses):
            yield response


