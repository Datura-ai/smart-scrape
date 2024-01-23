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

import time
import torch
import bittensor as bt
from typing import List, Union
from .config import RewardModelType, RewardScoringType
from .reward import BaseRewardModel, BaseRewardEvent
from utils.prompts import TwitterQuestionAnswerPrompt, TwitterSummaryLinksContetPrompt
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

def init_tokenizer(device):
    # https://huggingface.co/VMware/open-llama-7b-open-instruct
    # Fast tokenizer results in incorrect encoding, set the use_fast = False parameter.
    tokenizer = AutoTokenizer.from_pretrained(
        PromptRewardModel.reward_model_name, use_fast=False
    )
    # Generative default expects most recent token on right-hand side with padding on left.
    # https://github.com/huggingface/transformers/pull/10552
    tokenizer.padding_side = "left"

    # Check if the device is CPU or CUDA and set the precision accordingly
    torch_dtype = torch.float32 if device == 'cpu' else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        PromptRewardModel.reward_model_name, torch_dtype=torch_dtype
    ).to(device)
    return tokenizer, model

class PromptRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.prompt.value

    def __init__(self, device: str, scoring_type: None, tokenizer= None, model = None):
        super().__init__()
        self.device = device
        if not tokenizer:
            tokenizer, model = init_tokenizer(device)
            self.tokenizer = tokenizer
            self.model = model
        else:
            self.tokenizer = tokenizer
            self.model = model
    
        self.scoring_type = scoring_type

    def reward(self, prompt: str, response: bt.Synapse, name: str) -> BaseRewardEvent:
        try:
            completion = self.get_successful_completion(response=response)
            reward_event = BaseRewardEvent()

            with torch.no_grad():
                # Choose correct scoring prompt for request type.
                # Determine the scoring prompt based on the provided name or the default scoring type.
                scoring_prompt = None
                if self.scoring_type:
                    scoring_type = self.scoring_type
                else:
                    scoring_type = name

                scoring_prompt_text = None
                if scoring_type == RewardScoringType.twitter_question_answer_score:
                    scoring_prompt = TwitterQuestionAnswerPrompt()
                elif scoring_type == RewardScoringType.twitter_summary_links_content_template:
                    scoring_prompt = TwitterSummaryLinksContetPrompt()
                    # Convert list of links content to string before passing to the prompt
                    links_content_str = str(response.links_content)
                    scoring_prompt_text = scoring_prompt.text(completion, links_content_str)

                if scoring_prompt is None or not response.links_content:
                    reward_event.reward = 0
                    return reward_event

                if not scoring_prompt_text:
                    # Format scoring prompt for this completion.
                    scoring_prompt_text = scoring_prompt.text(prompt, completion)

                # Tokenize formatted scoring prompt.
                encodings_dict = self.tokenizer(
                    scoring_prompt_text,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encodings_dict["input_ids"].to(self.device)

                # Prompt local reward model.
                start_time = time.time()
                generated_tokens = self.model.generate(
                    input_ids, max_new_tokens=2, max_time=1
                )
                duration = time.time() - start_time
                generated_text = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                # Extract score from generated text.
                score_text = generated_text[0][len(scoring_prompt_text) :]
                score = scoring_prompt.extract_score(score_text)
                bt.logging.trace(
                    f"PromptRewardModel | {name} score: {score} | {repr(score_text)} | "
                    f"{duration:.2f}s | {repr(completion[:70])}"
                )
                if score == 0:
                    length = len(response.links_content) * 2 
                    score = length if length < 10 else 9
                # Scale 0-10 score to 0-1 range.
                score /= 10.0

                reward_event.reward = score
                return reward_event
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {e}")
            reward_event = BaseRewardEvent()
            reward_event.reward = 0
            return reward_event

    def get_rewards(
        self, prompt: str, responses: List[bt.Synapse], name: str, scoring_type: RewardScoringType = None
    ) -> List[BaseRewardEvent]:
        completions: List[str] = self.get_successful_completions(responses)
        bt.logging.debug(
            f"PromptRewardModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
        )
        bt.logging.trace(
            f"PromptRewardModel | prompt: {repr(prompt[:50])} ... {repr(prompt[-50:])}"
        )
        # Get all the reward results.
        reward_events = [
            self.reward(prompt, response, name) for response in responses
        ]

        return reward_events
