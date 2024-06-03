# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import traceback
import time
import torch
import bittensor as bt
import random
import asyncio
import re
import random
from typing import List
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
import torch
from typing import List
import numpy as np
import bittensor as bt


class PerformanceRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.twitter_content_relevance.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def calculate_percentile_rewards(
        self,
        process_times: List[float],
        max_reward: float = 1.0,
        min_reward: float = 0.1,
    ) -> List[float]:
        """
        Calculate rewards based on percentile ranking of process times.
        :param process_times: List of process times.
        :param max_reward: Maximum reward.
        :param min_reward: Minimum reward.
        :return: List of rewards corresponding to the process times.
        """
        # Convert process times to numpy array for efficient operations
        times_array = np.array(process_times)
        # Calculate percentiles for each time
        percentiles = 100 - np.argsort(np.argsort(times_array)) * 100.0 / (
            len(times_array) - 1
        )
        # Scale percentiles to the reward range
        scaled_rewards = min_reward + (max_reward - min_reward) * percentiles / 100
        return scaled_rewards.tolist()

    def get_rewards(
        self, prompt: str, responses: List[bt.Synapse], name: str, uids
    ) -> List[BaseRewardEvent]:
        reward_events = []
        try:
            # Initialize an empty list for process times
            process_times = []
            for response in responses:
                # Check if the response status code is 200 and there is a completion
                if response.dendrite.status_code == 200 and response.completion:
                    # If conditions are met, append the process time
                    process_times.append(response.dendrite.process_time)
                else:
                    # If conditions are not met, append a high process time to ensure a low reward
                    process_times.append(float("inf"))

            # Calculate rewards based on process times
            rewards = self.calculate_percentile_rewards(process_times)

            for response, reward in zip(responses, rewards):
                reward_event = BaseRewardEvent()
                # If the process time was set to infinity (for responses not meeting criteria), set reward to 0
                reward_event.reward = (
                    0 if response.dendrite.process_time == float("inf") else reward
                )
                reward_events.append(reward_event)

            # Assuming grouped_val_score_responses is calculated elsewhere or not needed for this example
            return reward_events, {}
        except Exception as e:
            error_message = f"Link Content Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
