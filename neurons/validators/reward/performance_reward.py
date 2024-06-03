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
import torch
import bittensor as bt
from typing import List, Tuple, Dict, Any, Union
import sys
import math
import copy
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import ScraperStreamingSynapse

# Assuming sturdy.constants and sturdy.utils.misc are available in your project
from neurons.validators.constants import QUERY_TIMEOUT, STEEPNESS, DIV_FACTOR, NUM_POOLS


class PerformanceRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.performance_score.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.is_default_normalization = False

    def get_response_times(self, uids: List[int], responses) -> Dict[int, float]:
        """
        Returns a dictionary of axons based on their response times.
        Adds a check for successful completion of the response.
        """
        axon_times = {
            uids[idx]: (
                response.dendrite.process_time
                if response.dendrite.process_time is not None
                and self.get_successful_completion(response)
                else QUERY_TIMEOUT
            )
            for idx, response in enumerate(responses)
        }
        return axon_times

    def sigmoid_scale(self, axon_time: float) -> float:
        """
        Scales the axon time using a sigmoid function.
        """
        offset = -float(NUM_POOLS) / DIV_FACTOR
        return (
            (1 / (1 + math.exp(STEEPNESS * axon_time + offset)))
            if axon_time < QUERY_TIMEOUT
            else 0
        )

    def reward(self, max_apy: float, miner_apy: float, axon_time: float) -> float:
        """
        Calculates the reward for a miner based on axon time and APY.
        """
        return (0.2 * self.sigmoid_scale(axon_time)) + (0.8 * miner_apy / max_apy)

    def get_rewards(
        self, prompt: str, responses: List[ScraperStreamingSynapse], name: str, uids
    ) -> Tuple[List[BaseRewardEvent]]:
        """
        Returns a list of reward events for the given responses.
        """
        reward_events = []
        try:
            axon_times = self.get_response_times(uids, responses)

            # Placeholder for APY calculations, assuming it's done elsewhere
            max_apy = max(miner_apy.values()) if miner_apy else 1.0
            miner_apy = {uid: 0.1 for uid in uids}  # Example static APY for each miner

            for uid in uids:
                reward_event = BaseRewardEvent()
                reward_event.reward = self.reward(
                    max_apy, miner_apy[uid], axon_times[uid]
                )
                reward_events.append(reward_event)

            return reward_events, {}
        except Exception as e:
            error_message = f"PerformanceRewardModel get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            for uid in uids:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
