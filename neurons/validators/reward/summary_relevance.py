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
import asyncio
import re
from typing import List, Union
from neurons.validators.reward.config import RewardModelType, RewardScoringType
from neurons.validators.reward.reward import BaseRewardModel, BaseRewardEvent
from neurons.validators.utils.prompts import SummaryRelevancePrompt, LinkContentPrompt, extract_score_and_explanation
from transformers import AutoTokenizer, AutoModelForCausalLM
from neurons.validators.utils import call_to_subnet_18_scoring, get_score_by_openai

from template.protocol import TwitterScraperStreaming, TwitterScraperTweet
from neurons.validators.reward.link_content_relevance import (
    init_tokenizer,
)
from enum import Enum

class ScoringSource(Enum):
    Subnet18 = 1
    OpenAI = 2
    LocalLLM = 3
        
class SummaryRelevanceRewardModel(BaseRewardModel):
    reward_model_name: str = "GTP-4"

    @property
    def name(self) -> str:
        return RewardModelType.summary_relavance_match.value

    def __init__(
        self,
        device: str,
        scoring_type: None,
        tokenizer=None,
        model=None,
        is_disable_tokenizer_reward=False,
    ):
        super().__init__()
        self.device = device

        self.scoring_type = scoring_type
        self.is_disable_tokenizer_reward = is_disable_tokenizer_reward
        if not is_disable_tokenizer_reward and tokenizer:
            self.tokenizer = tokenizer
            self.model = model

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
                links_content_str = str(response.links_content)
                scoring_prompt_text = scoring_prompt.text(
                    completion, links_content_str
                )

            if scoring_prompt is None or not response.links_content:
                return None

            if not scoring_prompt_text:
                # Format scoring prompt for this completion.
                scoring_prompt_text = scoring_prompt.text(prompt, completion)

            return scoring_prompt, [{"role": "user", "content": scoring_prompt_text}]
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {e}")
            return None

    def get_score_by_llm(self, messages):
        result = {}
        total_start_time = time.time()  # Start timing for total execution
        try:
            for message_dict in messages:  # Iterate over each dictionary in the list
                ((key, message_list),) = message_dict.items()

                with torch.no_grad():
                    # Choose correct scoring prompt for request type.
                    scoring_prompt_text = message_list[-1]["content"]  # Determine the scoring prompt based on the provided name or the default scoring type.

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
                        input_ids, max_new_tokens=500, max_time=5
                    )
                    duration = time.time() - start_time

                    # Decode the new tokens to get the generated text
                    generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                    # Extract score from generated text.
                    score_text = extract_score_and_explanation(generated_text)
                    bt.logging.info(f"Score text: {score_text}")
                    result[key] = score_text

            total_duration = time.time() - total_start_time  # Calculate total execution time
            bt.logging.info(f"Total execution time for get_score_by_llm: {total_duration} seconds")
        except Exception as e:
            bt.logging.error(f"Error in get_score_by_llm: {e}")
            return None
        return result

    def get_score_by_source(self, messages, source: ScoringSource):
        if source == ScoringSource.Subnet18:
            return call_to_subnet_18_scoring(messages)
        elif source == ScoringSource.OpenAI:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            return loop.run_until_complete(get_score_by_openai(messages=messages))
        else:
            return self.get_score_by_llm(messages=messages)

    def llm_processing(self, messages):
        # Initialize score_responses as an empty dictionary to hold the scoring results
        score_responses = {}

        # Define the order of scoring sources to be used
        scoring_sources = [
            ScoringSource.LocalLLM,  # Fallback to Local LLM if Subnet 18 fails or is disabled
            ScoringSource.LocalLLM,  # Fallback to Local LLM if Subnet 18 fails or is disabled
            ScoringSource.Subnet18,  # First attempt with Subnet 18
            ScoringSource.OpenAI,    # Final attempt with OpenAI if both Subnet 18 and Local LLM fail
        ]

        # Attempt to score messages using the defined sources in order
        for source in scoring_sources:
            # Attempt to score with the current source
            current_score_responses = self.get_score_by_source(messages=messages, source=source)
            if current_score_responses:
                # Update the score_responses with the new scores
                score_responses.update(current_score_responses)

                # Filter messages that still need scoring (i.e., messages that did not receive a score)
                messages = [msg for msg, score in current_score_responses.items() if not score]

                # If all messages have been scored, break out of the loop
                if not messages:
                    break
            else:
                bt.logging.info(f"Scoring with {source} failed or returned no results. Attempting next source.")
                
        return score_responses
    
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
                score_responses = self.llm_processing(messages)
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
            bt.logging.info(
                f"==================================Summary Relevance scoring Explanation Begins=================================="
            )
            for (index, response), uid_tensor in zip(enumerate(responses), uids):
                uid = uid_tensor.item()
                score = scores.get(str(index), 0)
                score_explain = score_text.get(str(index), "")
                reward_event = BaseRewardEvent()
                reward_event.reward = score
                reward_events.append(reward_event)
                bt.logging.info(
                    f"UID: {uid} | Score: {score:.2f} | Explanation: {score_explain.strip()}"
                )
                bt.logging.info(
                    f"----------------------------------------------------------------------"
                )
            bt.logging.info(
                f"==================================Summary Relevance Scoring Explanation Ends=================================="
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

from neurons.validators.reward.fail import completions_empty, prompt, completion_0

if __name__ == "__main__":
        tokenizer = None
        model = None
        device = 'cuda'
        tokenizer, model = init_tokenizer(device)
        summary = SummaryRelevanceRewardModel(
                    device=device,
                    scoring_type=RewardScoringType.summary_relevance_score_template,
                    tokenizer=tokenizer,
                    model=model,
                    is_disable_tokenizer_reward=False,
                )
        wallet = bt.wallet(name="validator-prod", hotkey="default")
        completions_empty
        scoring_messages = []
        completion_0.items()
        # merged_completions = {**completions_empty, **completion_0}
        for key, value in completion_0.items():
            # response = bt.dendrite(wallet=wallet)

            response = TwitterScraperStreaming(
                messages=prompt, model="", seed=1, is_intro_text=False
            )
            response.dendrite.status_code = 200
            response.completion = value
            response.links_content = [
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
        score_responses = summary.llm_processing(messages=messages)
        for key in score_responses.keys():
            llm_response = score_responses.get(key, "No response").replace("\n", " ")
            print(f" KEY: {key} ===========================================")
            print(f"{llm_response}")
           
            print(f"--------------------------------------------------------")

        print("Processing complete.")
