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
import traceback
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
)

from datura.protocol import ScraperStreamingSynapse
from neurons.validators.reward.reward_llm import RewardLLM
import json


class SummaryRelevanceRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.summary_relavance_match.value

    def __init__(self, device: str, scoring_type: None, llm_reward: RewardLLM):
        super().__init__()
        self.device = device
        self.reward_llm = llm_reward

        self.scoring_type = scoring_type

    def get_scoring_text(
        self, prompt: str, response: ScraperStreamingSynapse
    ) -> BaseRewardEvent:
        try:
            if "Twitter Search" in response.tools:
                completion = self.get_successful_twitter_completion(response=response)
            else:
                completion = self.get_successful_search_summary_completion(
                    response=response
                )

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

            # If tools include Twitter Search it scores summary of twitter, otherwise search
            is_twitter = "Twitter Search" in response.tools

            if (
                scoring_prompt is None
                or (is_twitter and not response.completion_links)
                or (not is_twitter and not response.search_completion_links)
            ):
                return None

            if not scoring_prompt_text:
                # Format scoring prompt for this completion.
                scoring_prompt_text = scoring_prompt.text(prompt, completion)

            return scoring_prompt, [
                {
                    "role": "system",
                    "content": scoring_prompt.get_system_message(is_twitter=is_twitter),
                },
                {"role": "user", "content": scoring_prompt_text},
            ]
        except Exception as e:
            bt.logging.error(f"Summary Relevance get_scoring_text: {str(e)}")
            return None

    def get_rewards(
        self, prompt: str, responses: List[ScraperStreamingSynapse], name: str, uids
    ) -> List[BaseRewardEvent]:
        try:
            completions: List[str] = self.get_successful_completions_for_summary(
                responses
            )
            bt.logging.info(f"SummaryRelevanceRewardModel | PROMPT: {prompt}")
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
            bt.logging.debug(
                f"SummaryRelevanceRewardModel | Calculating {len(filter_scoring_messages)} rewards (typically < 1 sec/reward)."
            )
            bt.logging.trace(
                f"SummaryRelevanceRewardModel | prompt: {repr(prompt[:50])} ... {repr(prompt[-50:])}"
            )
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
                bt.logging.info(
                    f"Executing llm_processing on {len(messages)} summary relevance messages."
                )
                score_responses = self.reward_llm.llm_processing(messages)
                if score_responses and isinstance(
                    score_responses, dict
                ):  # Ensure score_responses is a dictionary
                    for (key, score_result), (scoring_prompt, _) in zip(
                        score_responses.items(), filter_scoring_messages
                    ):
                        if (
                            score_result is not None
                        ):  # Check if score_result is not None
                            score = scoring_prompt.extract_score(score_result)
                            # Scale 0-10 score to 0-1 range.
                            score /= 10.0
                            scores[key] = score
                            score_text[key] = score_result

            # Iterate over responses and assign rewards based on scores
            reward_events = []

            # Initialize dictionaries to store zero and non-zero scores separately
            zero_scores = {}
            non_zero_scores = {}

            for (index, response), uid_tensor in zip(enumerate(responses), uids):
                uid = uid_tensor.item()
                score = scores.get(str(index), 0)
                score_explain = score_text.get(str(index), "")
                reward_event = BaseRewardEvent()
                reward_event.reward = score
                reward_events.append(reward_event)
                if score == 0:
                    zero_scores[uid] = score
                else:
                    non_zero_scores[uid] = score

            bt.logging.info(
                f"==================================Summary Relevance scoring Zero Scores  ({len(zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"==================================Summary Relevance scoring Non-Zero Scores ({len(non_zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(non_zero_scores))

            return reward_events, {}
        except Exception as e:
            error_message = f"Summary Relevanc Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
