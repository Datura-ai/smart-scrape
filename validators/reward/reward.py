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


class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    def get_rewards(
        self, prompt: str, completion: List[str], name: str
    ) -> Union[torch.FloatTensor, dict]:
        ...

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        self.count_limit = 3000

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        """
        This method normalizes the given rewards by updating the moving mean and variance statistics. The rewards are first standardized, and then scaled to the 0-1 range using a cumulative distribution function (CDF) to ensure they're in a comparable range across different environments.

        Args:
        rewards (torch.FloatTensor): The reward values to be normalized.

        Returns:
        torch.FloatTensor: The normalized reward values.

        Note:
        - This function uses Welford's online algorithm to update the mean and variance.
        - It standardizes the reward values using the updated mean and variance.
        - It then scales the standardized values to the 0-1 range using the error function (erf) as a CDF.
        """
        # Get the number of rewards (successful responses).
        new_count = rewards.numel()

        # Update stats only if there are new rewards.
        if 0 < new_count and 0 < self.count + new_count:
            # Calculate the mean and standard deviation of the new rewards.
            new_mean = rewards.mean()
            new_var = rewards.var(dim=0)

            # Compute the weights for the new and old rewards.
            new_weight = new_count / (self.count + new_count)
            old_weight = self.count / (self.count + new_count)

            # Save the difference in means before updating the old mean.
            diff = new_mean - self.mean

            # Update the old mean with the new mean and weights.
            self.mean = new_weight * new_mean + old_weight * self.mean
            # Update the old variance with the new variance and weights, and adjusting for the difference in means.
            self.var = (
                (new_weight * new_var)
                + (old_weight * self.var)
                + (new_weight * old_weight) * diff * diff
            )
            # Update the old count with the new count, but don't exceed the limit.
            self.count = min(self.count_limit, self.count + new_count)

        # Standardize the rewards using the updated mean and variance.
        rewards = rewards - self.mean
        if self.var > 0:
            rewards /= torch.sqrt(self.var)
        # Scale the standardized rewards to the range [0, 1] using the error function as a cumulative distribution function (CDF).
        rewards = 0.5 * (
            1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device))
        )

        return rewards

    def apply(
        self, prompt: str, responses: List[bt.Synapse], name: str
    ) -> Union[torch.FloatTensor, dict]:
        """Applies the reward model across each call. Unsuccessful responses are zeroed."""
        # Get indices of correctly responding calls.

        successful_completions_indices: List[int] = [
            idx
            for idx, resp in enumerate(responses)
            if resp.dendrite.status_code == 200
        ]

        # Get all completions from responding calls.
        successful_completions: List[str] = [
            responses[idx].completion.strip() for idx in successful_completions_indices
        ]

        # Reward each completion.
        reward_events = BaseRewardEvent.parse_reward_events(
            self.get_rewards(prompt, successful_completions, name)
        )
        successful_rewards = torch.tensor(
            reward_events.pop("reward"), dtype=torch.float32
        )

        # Softmax rewards across samples.
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)

        # Init zero rewards for all calls.
        filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

        # Fill reward tensor.
        for idx, reward, reward_normalized in zip(
            successful_completions_indices,
            successful_rewards,
            successful_rewards_normalized,
        ):
            filled_rewards[idx] = reward
            filled_rewards_normalized[idx] = reward_normalized

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
