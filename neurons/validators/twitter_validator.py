
import math
import torch
import wandb
import random
import json
import bittensor as bt
from base_validator import AbstractNeuron
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
from template.services.twitter import TwitterAPIClient
import asyncio

class TwitterScraperValidator:
    def __init__(self, neuron: AbstractNeuron):
        self.streaming = True
        self.query_type = "text"
        self.model = "gpt-4-1106-preview"
        self.weight = 1
        self.seed = 1234
        self.neuron = neuron
        self.timeout=75

        # Init device.
        bt.logging.debug("loading", "device")
        bt.logging.debug("self.neuron.config.neuron.device = ", str(self.neuron.config.neuron.device))

        self.reward_weights = torch.tensor(
            [
                self.neuron.config.reward.rlhf_weight,
                self.neuron.config.reward.prompt_based_weight,
                self.neuron.config.reward.dpo_weight,
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
            OpenAssistantRewardModel(device=self.neuron.config.neuron.device)
            if self.neuron.config.reward.rlhf_weight > 0
            else MockRewardModel(RewardModelType.rlhf.value),  

            PromptRewardModel(device=self.neuron.config.neuron.device)
            if self.neuron.config.reward.prompt_based_weight > 0
            else MockRewardModel(RewardModelType.prompt.value),

            # DirectPreferenceRewardModel(device=self.neuron.config.neuron.device)
            # if self.neuron.config.reward.dpo_weight > 0
            # else MockRewardModel(RewardModelType.prompt.value),                
        ]

        self.penalty_functions = [
            TaskValidationPenaltyModel(max_penalty=0.6),
            AccuracyPenaltyModel(max_penalty=1),
        ]

        self.twillio_api = TwitterAPIClient()
        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.neuron.metagraph.n)).to(self.neuron.config.neuron.device)
        bt.logging.debug(str(self.moving_averaged_scores))
    
    async def get_uids(self):
        available_uids = await self.neuron.get_available_uids()
        uid_list = list(available_uids.keys())
        uids = torch.tensor([random.choice(uid_list)]) if uid_list else torch.tensor([])
        bt.logging.info(" Random uids ---------- ", uids)
        uid_list = list(available_uids.keys())
        return uids.to(self.neuron.config.neuron.device)

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
    
    async def return_tokens(self, chunks):
        async for resp in chunks:
            if isinstance(resp, str):
                try:
                    chunk_data = json.loads(resp)
                    tokens = chunk_data.get("tokens", "")
                    bt.logging.trace(tokens)
                    yield tokens
                except json.JSONDecodeError:
                    bt.logging.trace(f"Failed to decode JSON chunk: {resp}")

    async def run_task_and_score(self, task: TwitterTask):
        task_name = task.task_name
        prompt = task.compose_prompt()

        bt.logging.debug("run_task", task_name)

        # Record event start time.
        event = {"name": task_name, "task_type": task.task_type}
        start_time = time.time()
        
        # Get random id on that step
        uids = await self.get_uids()
        axons = [self.neuron.metagraph.axons[uid] for uid in uids]
        synapse = TwitterScraperStreaming(messages=prompt, model=self.model, seed=self.seed)

        # Make calls to the network with the prompt.
        async_responses = await self.neuron.dendrite.forward(
            axons=axons,
            synapse=synapse,
            timeout=self.timeout,
            streaming=self.streaming,
            deserialize=False,
        )

        return async_responses, uids, event, start_time
    
    async def compute_rewards_and_penalties(self, event, prompt, task, responses, uids, start_time):
        try:
            bt.logging.info("Computing rewards and penalties")
            if responses:
                task.prompt_analysis = responses[0].prompt_analysis

            rewards = torch.zeros(len(responses), dtype=torch.float32).to(self.neuron.config.neuron.device)
            for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
                reward_i_normalized, reward_event = reward_fn_i.apply(task.base_text, responses, task.task_name)
                rewards += weight_i * reward_i_normalized.to(self.neuron.config.neuron.device)
                if not self.neuron.config.neuron.disable_log_rewards:
                    event = {**event, **reward_event}
                bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())
                bt.logging.info(f"Applied reward function: {reward_fn_i.name}")

            for penalty_fn_i in self.penalty_functions:
                raw_penalty_i, adjusted_penalty_i, applied_penalty_i = penalty_fn_i.apply_penalties(responses, task)
                rewards *= applied_penalty_i.to(self.neuron.config.neuron.device)
                if not self.neuron.config.neuron.disable_log_rewards:
                    event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
                    event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
                    event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()
                bt.logging.trace(str(penalty_fn_i.name), applied_penalty_i.tolist())
                bt.logging.info(f"Applied penalty function: {penalty_fn_i.name}")

            scattered_rewards = self.update_moving_averaged_scores(uids, rewards)
            self.log_event(task, event, start_time, uids, rewards, prompt=task.compose_prompt())

            scores = torch.zeros(len(self.neuron.metagraph.hotkeys))
            uid_scores_dict = {}
            wandb_data = {
                "modality": "twitter_scrapper",
                "prompts": {},
                "responses": {},
                "scores": {},
                "timestamps": {},
            }
            for uid_tensor, reward, response in zip(uids, rewards.tolist(), responses):
                uid = uid_tensor.item()  # Convert tensor to int
                uid_scores_dict[uid] = reward
                scores[uid] = reward  # Now 'uid' is an int, which is a valid key type
                wandb_data["scores"][uid] = reward
                wandb_data["responses"][uid] = response.completion
                wandb_data["prompts"][uid] = prompt
            bt.logging.info(f"Updated scores and wandb_data for uid: {uid}", wandb_data)
                    
            await self.neuron.update_scores(scores, wandb_data)

            return rewards, scattered_rewards
        except Exception as e:
            bt.logging.error(f"Error in compute_rewards_and_penalties: {e}")
            raise

    def update_moving_averaged_scores(self, uids, rewards):
        try:
            # uids = uids.to(self.neuron.config.neuron.device)
            # rewards = rewards.to(self.neuron.config.neuron.device)
            # scattered_rewards = self.moving_averaged_scores.scatter(0, uids, rewards)

            scattered_rewards = self.moving_averaged_scores.scatter(0, uids, rewards).to(self.neuron.config.neuron.device)
            bt.logging.info(f"Scattered reward: {torch.mean(scattered_rewards)}")

            alpha = self.neuron.config.neuron.moving_average_alpha
            self.moving_averaged_scores = alpha * scattered_rewards + (1 - alpha) * self.moving_averaged_scores.to(self.neuron.config.neuron.device)
            bt.logging.info(f"Moving averaged scores: {torch.mean(self.moving_averaged_scores)}")

            return scattered_rewards
        except Exception as e:
            bt.logging.error(f"Error in update_moving_averaged_scores: {e}")
            raise

    def log_event(self, task, event, start_time, uids, rewards, prompt):
        event.update({
            "step_length": time.time() - start_time,
            "prompt": prompt,
            "uids": uids.tolist(),
            "rewards": rewards.tolist(),
            "propmt": task.base_text
        })
        bt.logging.debug("Run Task event:", str(event))
    
    async def query_and_score(self):
        try:
            prompt = get_random_tweet_prompts(1)[0]

            task_name = "augment"
            task = TwitterTask(base_text=prompt, task_name=task_name, task_type="twitter_scraper", criteria=[])

            async_responses, uids, event, start_time = await self.run_task_and_score(
                task=task
            )
        
            responses = await self.process_async_responses(async_responses)
            if responses:
                task.prompt_analysis = responses[0].prompt_analysis
            await self.compute_rewards_and_penalties(event=event, 
                                                    prompt=prompt,
                                                    task=task, 
                                                    responses=responses, 
                                                    uids=uids, 
                                                    start_time=start_time)    
        except Exception as e:
            bt.logging.error(f"Error in query_and_score: {e}")
            raise
    
    
    async def organic(self, query):
        prompt = query['content']

        task_name = "augment"
        task = TwitterTask(base_text=prompt, task_name=task_name, task_type="twitter_scraper", criteria=[])

        async_responses, uids, event, start_time = await self.run_task_and_score(
            task=task
        )
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
                        tokens = chunk_data.get("tokens", "")
                        full_response += tokens
                        if "prompt_analysis" in chunk_data:
                            prompt_analysis_json = chunk_data["prompt_analysis"]
                            # Assuming prompt_analysis_json is a JSON string, parse it to a Python dict
                            prompt_analysis = json.loads(prompt_analysis_json)
                        yield tokens
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


        async def process_and_score_responses():
            if responses:
                task.prompt_analysis = responses[0].prompt_analysis
            await self.compute_rewards_and_penalties(event=event,
                                                     prompt=prompt, 
                                                     task=task, 
                                                     responses=responses, 
                                                     uids=uids, 
                                                     start_time=start_time)
            return responses  
        
        asyncio.create_task(process_and_score_responses())


