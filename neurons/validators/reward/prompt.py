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
import random
import requests
import os
from typing import List, Union
from .config import RewardModelType, RewardScoringType
from .reward import BaseRewardModel, BaseRewardEvent
from utils.prompts import TwitterQuestionAnswerPrompt, TwitterSummaryLinksContetPrompt
from transformers import AutoTokenizer, AutoModelForCausalLM


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

VALIDATOR_ACCESS_KEY = os.environ.get('VALIDATOR_ACCESS_KEY')
URL_SUBNET_18 = os.environ.get('URL_SUBNET_18')

def connect_to_subnet_18(data):
    headers = {
        "access_key": VALIDATOR_ACCESS_KEY,
        "Content-Type": "application/json"
    }
    response = requests.post(url=f"{URL_SUBNET_18}/scoring/", 
                             headers=headers, 
                             json=data)  # Using json parameter to automatically set the content-type to application/json

    if response.status_code in [401, 403]:
        bt.logging.error(f"Connection issue with Subnet 18: {response.text}")
        # os._exit(1)
    return response
    

class PromptRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.prompt.value

    def __init__(self, device: str, scoring_type: None, tokenizer= None, model = None, is_disable_tokenizer_reward=False):
        super().__init__()
        self.device = device

        # if not is_disable_tokenizer_reward:
        #     if not tokenizer:
        #         tokenizer, model = init_tokenizer(device)
        #         self.tokenizer = tokenizer
        #         self.model = model
        #     else:
        #         self.tokenizer = tokenizer
        #         self.model = model
            
        self.scoring_type = scoring_type
        self.is_disable_tokenizer_reward = is_disable_tokenizer_reward

    def get_scoring_text(self, prompt: str, response: bt.Synapse, name: str) -> BaseRewardEvent:
        try:
            completion = self.get_successful_completion(response=response)
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
                return None

            if not scoring_prompt_text:
                # Format scoring prompt for this completion.
                scoring_prompt_text = scoring_prompt.text(prompt, completion)

            return scoring_prompt, [{"role": "user", "content": scoring_prompt_text}]
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {e}")
            return None
            
    def get_rewards(
        self, prompt: str, responses: List[bt.Synapse], name: str, scoring_type: RewardScoringType = None
    ) -> List[BaseRewardEvent]:
        try:
            completions: List[str] = self.get_successful_completions(responses)
            bt.logging.debug(
                f"PromptRewardModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
            )
            bt.logging.trace(
                f"PromptRewardModel | prompt: {repr(prompt[:50])} ... {repr(prompt[-50:])}"
            )
            scoring_messages = [
                self.get_scoring_text(prompt, response, name) for response in responses
            ]
            filter_scoring_messages = [msg for msg in scoring_messages if msg is not None]
            # # Filter out None items from scoring_messages
            # messages = []
            # messages.extend({index: msg_content} for index, (_, msg_content) in enumerate(scoring_messages) if msg_content)
            messages = [{str(index): msg_content} for index, (_, msg_content) in enumerate(filter_scoring_messages)]
            print(messages)

            # messages = {
            #     { "118": [{"role": "user", "content": "test 1"}] },
            #     { "242": [{"role": "user", "content": "test 2"}] },
            #     { "244": [{"role": "user", "content": "test 3"}] }
            # }

            response = connect_to_subnet_18({
                    "messages": messages
            })
            if response.status_code != 200:
                raise Exception(response.text)
            
            score_responses = response.json()

            scores = {}
            for (key, score_result), (scoring_prompt, _) in zip(score_responses.items(), filter_scoring_messages):
                score = scoring_prompt.extract_score(score_result)
                # Scale 0-10 score to 0-1 range.
                score /= 10.0
                scores[key] = score
            
            # Iterate over responses and assign rewards based on scores
            reward_events = []
            for index, response in enumerate(responses):
                score = scores.get(str(index), 0)
                reward_event = BaseRewardEvent()
                reward_event.reward = score
                reward_events.append(reward_event)

            return reward_events
        except Exception as e:
            bt.logging.error(f"Reward model issue: {e}")
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event = 0
                reward_events.append(reward_event)
            return reward_events
