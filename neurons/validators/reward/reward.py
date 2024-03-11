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

import torch
import bittensor as bt
from typing import List, Union
from abc import abstractmethod
from dataclasses import dataclass, asdict, fields
from template.protocol import ScraperStreamingSynapse, TwitterScraperTweet
import re


@dataclass
class BaseRewardEvent:
    reward: float = 1.0
    normalized_reward: float = None

    @staticmethod
    def parse_reward_events(reward_events):
        if reward_events == None or len(reward_events) == 0:
            field_names = [field.name for field in fields(BaseRewardEvent())]
            empty_reward_event = dict(zip(field_names, [[]] * len(field_names)))
            return empty_reward_event

        field_names = [field.name for field in fields(reward_events[0])]
        reward_events = [
            asdict(reward_event).values() for reward_event in reward_events
        ]
        reward_event = dict(zip(field_names, list(zip(*reward_events))))
        return reward_event


pattern_to_check = r"<(?:Question|/Question|Answer|/Answer|Score|/Score)>|SM(?:[-_ ]SCS)?[-_ ]?(?:RDD|PNK|BLE|GRY|GRN)"


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> str: ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    def get_rewards(
        self, prompt: str, responses: List[ScraperStreamingSynapse], name: str, uids
    ) -> Union[torch.FloatTensor, dict]: ...

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        if self.var > 0:
            rewards /= torch.sqrt(self.var)

        common_formula = torch.erf(
            rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device)
        )
        rewards = torch.where(rewards == 0, 0, 0.5 * (1 + common_formula))

        return rewards

    def get_successful_completion(self, response: ScraperStreamingSynapse):
        # Check if the response is successful.
        if response.dendrite.status_code == 200 and response.completion_links:
            # Get the completion from the successful response.
            successful_completion = response.completion.strip()

            if re.search(pattern_to_check, successful_completion, flags=re.IGNORECASE):
                bt.logging.info(
                    f"Pattern validation issue Hotkey ID: {response.axon.hotkey}."
                )
                return None

            return successful_completion.strip()
        return None

    def get_successful_twitter_completion(self, response: ScraperStreamingSynapse):
        # Check if the response is successful.
        if response.dendrite.status_code == 200 and response.completion_links:
            # Get the completion from the successful response.
            successful_completion = response.get_twitter_completion().strip()

            if re.search(pattern_to_check, successful_completion, flags=re.IGNORECASE):
                bt.logging.info(
                    f"Pattern validation issue Hotkey ID: {response.axon.hotkey}."
                )
                return None

            return successful_completion.strip()
        return None

    def get_successful_search_summary_completion(
        self, response: ScraperStreamingSynapse
    ):
        # Check if the response is successful.
        search_completion = response.get_search_summary_completion()
        if response.dendrite.status_code == 200 and search_completion:
            # Get the completion from the successful response.
            successful_completion = search_completion.strip()

            if re.search(pattern_to_check, successful_completion, flags=re.IGNORECASE):
                bt.logging.info(
                    f"Pattern validation issue Hotkey ID: {response.axon.hotkey}."
                )
                return None

            return successful_completion.strip()
        return None

    def apply(
        self, prompt: str, responses: List[ScraperStreamingSynapse], name: str, uids
    ) -> Union[torch.FloatTensor, dict]:
        """Applies the reward model across each call. Unsuccessful responses are zeroed."""
        # Get indices of correctly responding calls.

        successful_completions_indices: List[int] = [
            idx
            for idx, resp in enumerate(responses)
            if resp.dendrite.status_code == 200 and resp.completion_links
        ]

        # Reward each completion.
        reward_events = BaseRewardEvent.parse_reward_events(
            self.get_rewards(prompt, responses, name, uids)
        )
        successful_rewards = reward_events
        successful_rewards = torch.tensor(
            reward_events.pop("reward"), dtype=torch.float32
        )

        # Softmax rewards across samples.
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)

        # Init zero rewards for all calls.
        filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

        for idx in successful_completions_indices:
            filled_rewards[idx] = successful_rewards[idx]
            filled_rewards_normalized[idx] = successful_rewards_normalized[idx]

        # Fill every item of the reward_events
        for name, reward_values in reward_events.items():
            filled_values = [None] * len(responses)
            for idx, reward_value in zip(successful_completions_indices, reward_values):
                filled_values[idx] = reward_value
            reward_events[name] = filled_values

        # Name each item of the reward event with the reward model name.
        reward_events = {f"{self.name}_{k}": v for k, v in reward_events.items()}
        reward_events[self.name] = filled_rewards.tolist()
        reward_events[self.name + "_normalized"] = filled_rewards_normalized.tolist()

        # Warns unexpected behavior for rewards
        if torch.isnan(filled_rewards_normalized).any():
            bt.logging.warning(
                f"The tensor from {self.name} contains NaN values: {filled_rewards_normalized}"
            )
            filled_rewards_normalized = filled_rewards_normalized.nan_to_num_(nan=0.0)

        # Return the filled rewards.
        return filled_rewards_normalized, reward_events
