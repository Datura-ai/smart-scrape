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
import traceback
import time
import torch
import bittensor as bt
import random
import asyncio
import re
import html
import random
from typing import List, Union
from .config import RewardModelType, RewardScoringType
from .reward import BaseRewardModel, BaseRewardEvent, pattern_to_check
from neurons.validators.utils.prompts import (
    SummaryRelevancePrompt,
    LinkContentPrompt,
    extract_score_and_explanation,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from template.protocol import ScraperStreamingSynapse, TwitterScraperTweet
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor
from template.services.twitter_api_wrapper import TwitterAPIClient
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.prompts import ScoringPrompt
import json
from datetime import datetime
import pytz


class LinkContentRelevanceModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.link_content_match.value

    def __init__(self, device: str, scoring_type: None, llm_reward: RewardLLM):
        super().__init__()
        self.device = device
        self.reward_llm = llm_reward

        self.scoring_type = scoring_type
        self.tw_client = TwitterAPIClient()

    async def llm_process_validator_tweets(self, prompt, tweets_list):
        scoring_messages = []
        for tweet in tweets_list:
            val_text = tweet.full_text
            val_tweet_id = tweet.id
            result = self.get_scoring_text(
                prompt=prompt, content=val_text, response=None
            )
            if result:
                scoring_prompt, scoring_text = result
                scoring_messages.append({str(val_tweet_id): scoring_text})
        score_responses = self.reward_llm.llm_processing(scoring_messages)
        return score_responses

    async def process_tweets(self, prompt, responses):
        try:
            non_fetched_links = {}
            start_time = time.time()
            all_links = [
                random.choice(response.completion_links)
                for response in responses
                if response.completion_links
            ]
            unique_links = list(
                set(all_links)
            )  # Remove duplicates to avoid redundant tasks

            if len(unique_links) == 0:
                bt.logging.info("No unique links found to process.")
                return
            tweets_list = await TwitterScraperActor().get_tweets(urls=unique_links)
            for response in responses:
                ids = [
                    self.tw_client.extract_tweet_id(link)
                    for link in response.completion_links
                ]

                for tweet in tweets_list:
                    if tweet.id in ids:
                        response.validator_tweets.append(tweet)
            if len(unique_links) == 0:
                bt.logging.info("No unique links found to process.")
                return {}

            val_score_responses = await self.llm_process_validator_tweets(
                prompt, tweets_list
            )
            end_time = time.time()
            bt.logging.info(
                f"Fetched Twitter links method took {end_time - start_time} seconds. "
                f"All links count: {len(all_links)}, Unique links count: {len(unique_links)}, "
                f"APIFY fetched tweets links count: {len(tweets_list)}"
            )
            fetched_tweet_ids = {tweet.id for tweet in tweets_list}
            non_fetched_links = [
                link
                for link in unique_links
                if self.tw_client.extract_tweet_id(link) not in fetched_tweet_ids
            ]

            bt.logging.info(
                f"Twitter Links not fetched Amount: {len(non_fetched_links)}; List: {non_fetched_links}; For prompt: [{prompt}]"
            )
            if len(non_fetched_links):
                bt.logging.info(
                    f"Unique Twitter Links Amount: {len(unique_links)}; List: {unique_links};"
                )
            return val_score_responses
        except Exception as e:
            bt.logging.error(f"Error in process_tweets: {str(e)}")
            return {}

    def format_text_for_match(self, text):
        # url shorteners can cause problems with tweet verification, so remove urls from the text comparison.
        text = re.sub(r"(https?://)?\S+\.\S+\/?(\S+)?", "", text)
        # Some scrapers put the mentions at the front of the text, remove them.
        text = re.sub(r"^(@\w+\s*)+", "", text)
        # And some trim trailing whitespace at the end of newlines, so ignore whitespace.
        text = re.sub(r"\s+", "", text)
        # And some have special characters escaped as html entities
        text = html.unescape(text)
        # The validator apify actor uses the tweet.text field and not the note_tweet field (> 280) charts, so only
        # use the first 280 chars for comparison.
        text = text[:280]
        return text

    def check_response_random_tweet(self, response: ScraperStreamingSynapse):
        try:
            tweet_score = 0

            completion = self.get_successful_completion(response=response)
            if not completion:
                return 0

            miner_tweets = response.miner_tweets

            miner_tweets_data = miner_tweets.get("data", [])
            # miner_tweets_meta = miner_tweets.get('meta', {})
            # miner_tweets_users = miner_tweets.get('includes', {}).get('users', [])
            miner_tweets_amount = miner_tweets.get("meta", {}).get("result_count", 0)

            # Assign completion links and validator tweets from the response
            completion_links = response.completion_links

            if len(completion_links) < 2:
                # at least miners should provide two twitter links
                return 0

            # Check if there are no completion links, no miner tweets, or no validator tweets
            if (
                not completion_links
                or miner_tweets_amount == 0
                or not response.validator_tweets
            ):
                return 0

            # Select a random tweet from the validator's tweets
            val_tweet = random.choice(response.validator_tweets)

            # Extract content, ID, and creation time of the validator tweet
            val_tweet_content = val_tweet.full_text
            val_tweet_id = val_tweet.id
            val_tweet_created_at = val_tweet.created_at
            # Find the corresponding miner tweet by ID
            miner_tweet = next(
                (tweet for tweet in miner_tweets_data if tweet["id"] == val_tweet_id),
                None,
            )
            # If there is no corresponding miner tweet, append a score of 0
            if miner_tweet:
                # If a corresponding miner tweet is found, extract its creation time and text

                miner_tweet_text = miner_tweet["text"]

                if not miner_tweet_text or re.search(
                    pattern_to_check, miner_tweet_text, flags=re.IGNORECASE
                ):
                    return 0

                # Prepare texts for comparison by normalizing them
                miner_text_compared = self.format_text_for_match(miner_tweet_text)
                validator_text_compared = self.format_text_for_match(val_tweet_content)
                # Compare the normalized texts and creation times of the tweets

                if miner_text_compared == validator_text_compared:
                    tweet_score = 1.5

                converted_val_tweet_created_at = (
                    datetime.strptime(val_tweet_created_at, "%a %b %d %H:%M:%S %z %Y")
                    .astimezone(pytz.UTC)
                    .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                    + "Z"
                )

                if miner_tweet.get("created_at") == converted_val_tweet_created_at:
                    # If both match, append a score of 1
                    tweet_score += 0.5

                if tweet_score == 0:
                    # If there is a discrepancy, log the details and append a score of 0
                    bt.logging.info(
                        f"Miner tweet - Created at: {miner_tweet.get('created_at', 'None')}, Text: {miner_text_compared}"
                    )
                    bt.logging.debug(
                        f"Validator tweet - Created at: {val_tweet_created_at}, Text: {validator_text_compared}"
                    )

            else:
                bt.logging.debug(
                    "No corresponding miner tweet found for the validator tweet."
                )
                tweet_score = 0
            return tweet_score
        except Exception as e:
            bt.logging.error(f"check_response_random_tweet: {str(e)}")
            return 0

    def get_scoring_text(
        self, prompt: str, content: str, response: bt.Synapse
    ) -> BaseRewardEvent:
        try:
            if response:
                completion = self.get_successful_completion(response=response)
                if not completion:
                    return None

            scoring_prompt = None

            scoring_prompt_text = None

            scoring_prompt = LinkContentPrompt()

            if not scoring_prompt_text:
                scoring_prompt_text = scoring_prompt.text(prompt, content)

            return scoring_prompt, [
                {"role": "user", "content": scoring_prompt_text},
                {"role": "system", "content": scoring_prompt.get_system_message()},
            ]
        except Exception as e:
            bt.logging.error(f"Error in Prompt reward method: {str(e)}")
            return None

    def get_rewards(
        self, prompt: str, responses: List[bt.Synapse], name: str, uids
    ) -> List[BaseRewardEvent]:
        try:
            completions: List[str] = self.get_successful_completions(responses)
            bt.logging.debug(
                f"LinkContentRelevanceModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
            )
            bt.logging.trace(
                f"LinkContentRelevanceModel | prompt: {repr(prompt[:50])} ... {repr(prompt[-50:])}"
            )

            val_score_responses = asyncio.get_event_loop().run_until_complete(
                self.process_tweets(prompt=prompt, responses=responses)
            )
            bt.logging.info(f"LinkContentRelevanceModel | PROMPT: {prompt}")
            bt.logging.info(
                f"LinkContentRelevanceModel | Keys in val_score_responses: {len(val_score_responses.keys()) if val_score_responses else 'No val_score_responses available'}"
            )
            scores = [
                self.check_response_random_tweet(response) for response in responses
            ]

            reward_events = []
            scoring_messages = []
            for apify_score, response, uid_tensor in zip(
                scores, responses, uids
            ):  # Fixed variable name from 'response' to 'responses'
                # bt.logging.info(f"Processing score for response with miner tweets.")
                uid = uid_tensor.item()
                miner_tweets = response.miner_tweets
                miner_tweets_data = miner_tweets.get("data", [])

                if not len(response.validator_tweets):
                    for link in random.sample(
                        response.completion_links,
                        (
                            1
                            if len(response.completion_links) > 1
                            else len(response.completion_links)
                        ),
                    ):
                        tweet_id = self.tw_client.extract_tweet_id(link)
                        miner_tweet = next(
                            (
                                tweet
                                for tweet in miner_tweets_data
                                if tweet["id"] == tweet_id
                            ),
                            None,
                        )
                        if miner_tweet:
                            miner_tweet_text = miner_tweet["text"]
                            if miner_tweet_text:
                                result = self.get_scoring_text(
                                    prompt, miner_tweet_text, response
                                )
                                if result:
                                    scoring_prompt, scoring_text = result
                                    scoring_messages.append({str(uid): scoring_text})

            bt.logging.info(
                f"Executing llm_processing on {len(scoring_messages)} Link Content Relevance messages."
            )
            score_responses = self.reward_llm.llm_processing(scoring_messages)
            reward_events = []
            scoring_prompt = ScoringPrompt()
            for apify_score, response, uid_tensor in zip(
                scores, responses, uids
            ):  # Fixed variable name from 'response' to 'responses'
                uid = uid_tensor.item()
                reward_event = BaseRewardEvent()
                reward_event.reward = 0

                score_result = None
                if len(response.validator_tweets):
                    val_tweet = random.choice(response.validator_tweets)
                    val_tweet_id = val_tweet.id
                    if val_score_responses:
                        score_result = val_score_responses.get(str(val_tweet_id), None)
                else:
                    if score_responses:
                        score_result = score_responses.get(str(uid), None)
                if score_result is None:
                    bt.logging.info(
                        f"Link Content Relevance get_rewards: No score response for UID '{uid}'"
                    )
                    score = 0  # Default score or another logic to handle missing scores
                else:
                    score = scoring_prompt.extract_score(score_result)
                    score /= 10.0
                    reward_event.reward = score
                if apify_score:
                    reward_event.reward = min(reward_event.reward * apify_score, 1)
                else:
                    reward_event.reward /= 3
                reward_events.append(reward_event)

                zero_scores = {}
                non_zero_scores = {}

            for (index, response), uid_tensor, reward_e in zip(
                enumerate(responses), uids, reward_events
            ):
                uid = uid_tensor.item()
                if reward_e.reward == 0:
                    # score_explain = score_responses.get(str(uid), "")
                    zero_scores[uid] = reward_e.reward
                else:
                    non_zero_scores[uid] = reward_e.reward

            bt.logging.info(
                f"==================================Links Content scoring Zero Scores  ({len(zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"==================================Links Content scoring Non-Zero Scores ({len(non_zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(non_zero_scores))
            return reward_events
        except Exception as e:
            error_message = f"Link Content Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events
