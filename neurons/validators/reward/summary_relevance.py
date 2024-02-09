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
from typing import List, Union
from .config import RewardModelType, RewardScoringType
from .reward import BaseRewardModel, BaseRewardEvent
from utils.prompts import SummaryRelevancePrompt, LinkContentPrompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from neurons.validators.utils import call_to_subnet_18_scoring
from template.utils import call_openai



class SummaryRelevanceRewardModel(BaseRewardModel):
    reward_model_name: str = "GTP-4"

    @property
    def name(self) -> str:
        return RewardModelType.prompt.value

    def __init__(self, device: str, scoring_type: None):
        super().__init__()
        self.device = device

        self.scoring_type = scoring_type

    def get_scoring_text(self, prompt: str, response: bt.Synapse) -> BaseRewardEvent:
        try:
            completion = self.get_successful_completion(response=response)
            if not self.scoring_type:
                return None
            # Choose correct scoring prompt for request type.
            # Determine the scoring prompt based on the provided name or the default scoring type.
            scoring_prompt = None

            scoring_prompt_text = None
            if self.scoring_type.value == RewardScoringType.summary_relevance_score_template.value:
                scoring_prompt = SummaryRelevancePrompt()
            elif self.scoring_type.value == RewardScoringType.link_content_relevance_template.value:
                scoring_prompt = LinkContentPrompt()
                # Convert list of links content to string before passing to the prompt
                completion_links_str = str(response.completion_links)
                scoring_prompt_text = scoring_prompt.text(completion, completion_links_str)

            if scoring_prompt is None or not response.completion_links:
                return None

            if not scoring_prompt_text:
                # Format scoring prompt for this completion.
                scoring_prompt_text = scoring_prompt.text(prompt, completion)

            return scoring_prompt, [{"role": "user", "content": scoring_prompt_text}]
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {e}")
            return None
        
    async def send_messages_to_openai(self, messages):
        query_tasks = []
        for message_dict in messages:  # Iterate over each dictionary in the list
            (key, message_list), = message_dict.items()
            
            async def query_openai(message):
                try:
                    return await call_openai(
                        messages=message, 
                        temperature=0.2,
                        model='gpt-3.5-turbo-16k',
                    )
                except Exception as e:
                    print(f"Error sending message to OpenAI: {e}")
                    return ""  # Return an empty string to indicate failure

            task = query_openai(message_list)
            query_tasks.append(task)

        query_responses = await asyncio.gather(*query_tasks, return_exceptions=True)

        result = {}
        for response, message_dict in zip(query_responses, messages):
            if isinstance(response, Exception):
                print(f"Query failed with exception: {response}")
                response = ""  # Replace the exception with an empty string in the result
            (key, message_list), = message_dict.items()
            result[key] = response
        return result
            
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
            filter_scoring_messages = [msg for msg in scoring_messages if msg is not None]
            # # Filter out None items from scoring_messages
            # messages = []
            # messages.extend({index: msg_content} for index, (_, msg_content) in enumerate(scoring_messages) if msg_content)
            # messages = [{str(index): msg_content} for index, (_, msg_content) in enumerate(filter_scoring_messages)]
            messages = [{str(index): item[1]} for index, item in enumerate(scoring_messages) if item is not None]

            scores = {}
            score_text = {}
            if messages:
                response = call_to_subnet_18_scoring({
                    "messages": messages
                })
                if response.status_code != 200:
                    bt.logging.error(f"ERROR connect to Subnet 18: {response.text}")
                    loop = asyncio.get_event_loop_policy().get_event_loop()
                    score_responses = loop.run_until_complete(self.send_messages_to_openai(messages=messages))
                else:
                    score_responses = response.json()

                for (key, score_result), (scoring_prompt, _) in zip(score_responses.items(), filter_scoring_messages):
                    score = scoring_prompt.extract_score(score_result)
                    # Scale 0-10 score to 0-1 range.
                    score /= 10.0
                    scores[key] = score
                    score_text[key] = score_result
            
            # Iterate over responses and assign rewards based on scores
            reward_events = []
            bt.logging.info(f"==================================Scoring Explanation Begins==================================")
            for (index, response), uid_tensor in zip(enumerate(responses), uids):
                uid = uid_tensor.item()
                score = scores.get(str(index), 0)
                score_explain = score_text.get(str(index), '')
                reward_event = BaseRewardEvent()
                reward_event.reward = score
                reward_events.append(reward_event)
                bt.logging.info(f"UID: {uid} | Score: {score:.2f} | Explanation: {score_explain.strip()}")
                bt.logging.info(f"----------------------------------------------------------------------")
            bt.logging.info(f"==================================Scoring Explanation Ends==================================")

            return reward_events
        except Exception as e:
            bt.logging.error(f"Reward model issue: {e}")
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events
