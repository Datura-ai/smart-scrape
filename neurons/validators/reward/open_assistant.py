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
from typing import List, Union
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
import bittensor as bt
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class OpenAssistantRewardModel(BaseRewardModel):
    #More details here: https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"

    @property
    def name(self) -> str:
        return RewardModelType.rlhf.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            OpenAssistantRewardModel.reward_model_name
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            OpenAssistantRewardModel.reward_model_name
        ).to(self.device)

    def reward_single(self, prompt: str, completion: str, name: str) -> BaseRewardEvent:
        try:
            reward_event = BaseRewardEvent()

            with torch.no_grad():
                inputs = self.tokenizer(prompt, completion, return_tensors="pt").to(
                    self.device
                )
                reward_event.reward = float(self.model(**inputs).logits[0].cpu().detach())
                return reward_event
        except Exception as e:
            bt.logging.error(f"Error in  OpenAssistantRewardModel reward_single method: {e}")
    def get_rewards(
        self, prompt: str, responses: List[bt.Synapse], name: str
    ) -> List[BaseRewardEvent]:
        completions: List[str] = self.get_successful_completions(responses)
        # Get all the reward results.
        reward_events = [
            self.reward_single(prompt, completion, name) for completion in completions
        ]

        return reward_events
