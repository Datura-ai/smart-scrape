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
# DEALINGS IN THE SOFTWARE.p

import time
import torch
import bittensor as bt
import random
import asyncio
import re
from typing import List, Union
from neurons.validators.reward.config import RewardModelType, RewardScoringType
from neurons.validators.reward.reward import BaseRewardModel, BaseRewardEvent
from neurons.validators.utils.prompts import (
    SummaryRelevancePrompt,
    LinkContentPrompt,
    extract_score_and_explanation,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from template.protocol import ScraperStreamingSynapse, TwitterScraperTweet
from neurons.validators.reward.reward_llm import RewardLLM


class SummaryRelevanceRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.summary_relavance_match.value

    def __init__(
        self,
        device: str,
        scoring_type: None,
        llm_reward : RewardLLM
    ):
        super().__init__()
        self.device = device
        self.reward_llm = llm_reward

        self.scoring_type = scoring_type

    def get_scoring_text(self, prompt: str, response: bt.Synapse) -> BaseRewardEvent:
        try:
            completion = self.get_successful_completion(response=response)

            if not completion:
                return None

            if not self.scoring_type:
                return None
            # Choose correct scoring prompt for request type.
            # Determine the scoring prompt based on the provided name or the default scoring type.
            scoring_prompt = None

            scoring_prompt_text = None
            if (
                self.scoring_type.value
                == RewardScoringType.summary_relevance_score_template.value
            ):
                scoring_prompt = SummaryRelevancePrompt()
            elif (
                self.scoring_type.value
                == RewardScoringType.link_content_relevance_template.value
            ):
                scoring_prompt = LinkContentPrompt()
                # Convert list of links content to string before passing to the prompt
                completion_links_str = str(response.completion_links)
                scoring_prompt_text = scoring_prompt.text(
                    completion, completion_links_str
                )

            if scoring_prompt is None or not response.completion_links:
                return None

            if not scoring_prompt_text:
                # Format scoring prompt for this completion.
                scoring_prompt_text = scoring_prompt.text(prompt, completion)

            return scoring_prompt, [{"role": "user", "content": scoring_prompt_text}]
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {e}")
            return None
        
    def get_rewards(
        self, prompt: str, responses: List[bt.Synapse], name: str, uids
    ) -> List[BaseRewardEvent]:
        try:
            completions: List[str] = self.get_successful_completions(responses)
            bt.logging.debug(
                f"SummaryRelevanceRewardModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
            )
            bt.logging.trace(
                f"SummaryRelevanceRewardModel | prompt: {repr(prompt[:50])} ... {repr(prompt[-50:])}"
            )
            scoring_messages = [
                self.get_scoring_text(prompt, response) for response in responses
            ]
            filter_scoring_messages = [
                msg for msg in scoring_messages if msg is not None
            ]
            # # Filter out None items from scoring_messages
            # messages = []
            # messages.extend({index: msg_content} for index, (_, msg_content) in enumerate(scoring_messages) if msg_content)
            # messages = [{str(index): msg_content} for index, (_, msg_content) in enumerate(filter_scoring_messages)]
            messages = [
                {str(index): item[1]}
                for index, item in enumerate(scoring_messages)
                if item is not None
            ]

            scores = {}
            score_text = {}
            if messages:
                score_responses = self.reward_llm.llm_processing(messages)
                if score_responses:
                    for (key, score_result), (scoring_prompt, _) in zip(
                        score_responses.items(), filter_scoring_messages
                    ):
                        score = scoring_prompt.extract_score(score_result)
                        # Scale 0-10 score to 0-1 range.
                        score /= 10.0
                        scores[key] = score
                        score_text[key] = score_result

            # Iterate over responses and assign rewards based on scores
            reward_events = []
           
            for (index, response), uid_tensor in zip(enumerate(responses), uids):
                uid = uid_tensor.item()
                if score == 0:
                    bt.logging.info(
                        f"==================================Summary Relevance scoring Explanation Begins=================================="
                    )
                    bt.logging.info(
                        f"PROMPT: {prompt}"
                    )
                    bt.logging.info(
                        f"----------------------------------------------------------------------------------------------"
                    )
                    score = scores.get(str(index), 0)
                    score_explain = score_text.get(str(index), "")
                    reward_event = BaseRewardEvent()
                    reward_event.reward = score
                    reward_events.append(reward_event)
                    bt.logging.info(
                        f"UID: {uid} | Score: {score:.2f} | Explanation: {score_explain.strip()}"
                    )
                    bt.logging.info(
                        f"==================================Summary Relevance Scoring Explanation Ends=================================="
                    )
                else:
                    bt.logging.info(
                        f"UID: {uid} | Score: {score:.2f}"
                    )

            return reward_events
        except Exception as e:
            bt.logging.error(f"Reward model issue: {e}")
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events


if __name__ == "__main__":
    from neurons.validators.reward.test_summary import completions_empty, prompt, completion_0, completion_1

    tokenizer = None
    model = None
    device = "cuda"
    llm = RewardLLM()
    llm.init_pipe_zephyr()
    summary = SummaryRelevanceRewardModel(
        device=device,
        scoring_type=RewardScoringType.summary_relevance_score_template,
        llm_reward=llm
    )
    wallet = bt.wallet(name="validator-prod", hotkey="default")
    completions_empty
    scoring_messages = []
    completion_0.items()
    # merged_completions = {**completions_empty, **completion_0}
    for key, value in completion_1.items():
        # response = bt.dendrite(wallet=wallet)

        response = ScraperStreamingSynapse(
            messages=prompt, model="", seed=1, is_intro_text=False
        )
        response.dendrite.status_code = 200
        response.completion = value
        response.completion_links = [
            "https://twitter.com/Carlossainz55/status/1753134900129956343",
            "https://twitter.com/Carlossainz55/status/1753134900129956343",
        ]
        result = summary.get_scoring_text(prompt, response)
        scoring_messages.append(result)

    messages = [
        {str(index): item[1]}
        for index, item in enumerate(scoring_messages)
        if item is not None
    ]
    score_responses = summary.reward_llm.llm_processing(messages=messages)
    for key in score_responses.keys():
        llm_response = score_responses.get(key, "No response").replace("\n", " ")
        print(f" KEY: {key} ===========================================")
        print(f"{llm_response}")

        print(f"--------------------------------------------------------")

    print("Processing complete.")
