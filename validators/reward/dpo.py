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
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    NoRepeatNGramLogitsProcessor,
)


class DirectPreferenceRewardModel(BaseRewardModel):
    reward_model_name: str = "cerebras/btlm-3b-8k-base"

    @property
    def name(self) -> str:
        return RewardModelType.dpo.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.penalty = 1.2  # Same penalty as the original [paper](https://arxiv.org/pdf/1909.05858.pdf).
        self.tokenizer = AutoTokenizer.from_pretrained(
            DirectPreferenceRewardModel.reward_model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            DirectPreferenceRewardModel.reward_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.ngram_logit_processor = NoRepeatNGramLogitsProcessor(ngram_size=5)

    def reward_single(
        self, prompt: str, completion: str, name: str, with_penalty=True
    ) -> BaseRewardEvent:
        r"""Calculates a direct preference optimization (DPO) style reward for a completion,
        which is a reference model's average log-probability for completion tokens given a prompt.
        Uses guidance from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py.
        """

        reward_event = BaseRewardEvent()

        with torch.no_grad():
            # Check if completion is
            if completion.strip() == "" or len(completion) <= 5:
                # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)
                reward_event.reward = -11.0
                return reward_event

            # Tokenize the combined prompt + completion.
            combined = (
                self.tokenizer(prompt + completion, return_tensors="pt")
                .input_ids[0]
                .to(self.device)
            )  # [seq_len]
            # Tokenize only the prompt, to help determine prompt token length.
            prompt_part = (
                self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
            )  # [prompt_len]

            # Completion doesn't fit into model sequence, so return lowest reward.
            if self.tokenizer.model_max_length <= len(prompt_part):
                reward_event.reward = -11.0
                return reward_event

            # Truncate combined to fit into model max sequence length.
            if self.tokenizer.model_max_length < len(combined):
                combined = combined[: self.tokenizer.model_max_length]

            labels = combined.clone()  # [seq_len]
            # Ignore prompt part for calculating reward.
            labels[: len(prompt_part)] = -100
            # Label only each next token prediction ground-truth.
            labels = labels[1:]  # [seq_len-1]
            loss_mask = labels != -100  # [seq_len-1]
            # Dummy token to allow for indexing, but loss will be ignored.
            labels[labels == -100] = 0
            # Reshape for gather operation.
            labels = labels.unsqueeze(0).unsqueeze(2)  # [batch_size=1, seq_len-1, :]

            # Forward pass to calculate logit predictions for each sequence position.
            logits = self.model(
                combined.unsqueeze(0)
            ).logits  # [batch_size=1, seq_len, vocab_len]
            # Predict only where labels are available.
            logits = logits[:, :-1, :]  # [batch_size=1, seq_len-1, vocab_len]

            if with_penalty:
                org_logit = logits.clone()
                logits = self.ngram_logit_processor(
                    combined[len(prompt_part) :].reshape(1, -1).clone(),
                    logits.permute(0, 2, 1),
                ).permute(0, 2, 1)
                # ngram_logit_processor set punished tokens to -inf, resetting them to 10 std below instead
                logits[logits == -float("Inf")] = (
                    org_logit.mean() - org_logit.std() * 10
                )

            # Rescale via log(softmax(logits)).
            logits = logits.log_softmax(-1)
            # Calculate the model's log-probability for each actual completion token.
            per_token_logps = torch.gather(logits, dim=2, index=labels).squeeze(
                2
            )  # [batch_size=1, seq_len-1]
            # Average log-probability over completion sequence.
            reward = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(
                -1
            )  # [batch_size=1]
            reward = reward[0].cpu().detach()

            # NaNs can possibly arise through log(0)=-inf, replace with suitably small logits.
            if torch.isnan(reward) or torch.isinf(reward):
                reward_event.reward = -11
                return reward_event

            reward_event.reward = reward.item()
            return reward_event

    def get_rewards(
        self, prompt: str, completions: List[str], name: str
    ) -> List[BaseRewardEvent]:
        # Get all the reward results.
        reward_events = [
            self.reward_single(prompt, completion, name) for completion in completions
        ]

        bt.logging.trace(
            f"DirectPreferenceRewardModel | rewards: {[reward_event.reward for reward_event in reward_events]}"
        )

        return reward_events
