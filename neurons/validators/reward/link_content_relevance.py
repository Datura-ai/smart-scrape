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
import html
import random
from typing import List, Union
from .config import RewardModelType, RewardScoringType
from .reward import BaseRewardModel, BaseRewardEvent
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

    async def process_tweets(self, responses):
        try:
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
            end_time = time.time()
            bt.logging.info(
                f"Fetched Twitter links method took {end_time - start_time} seconds. "
                f"All links count: {len(all_links)}, Unique links count: {len(unique_links)}, "
                f"APIFY fetched tweets links count: {len(tweets_list)}"
            )
        except Exception as e:
            bt.logging.error(f"Error in process_tweets: {str(e)}")
            return

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
            validator_tweets = response.validator_tweets

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
                # Prepare texts for comparison by normalizing them
                miner_text_compared = self.format_text_for_match(miner_tweet_text)
                validator_text_compared = self.format_text_for_match(val_tweet_content)
                # Compare the normalized texts and creation times of the tweets

                if (
                    miner_text_compared == validator_text_compared
                ):
                    tweet_score = 1.5
                    
                if (
                    miner_tweet.get("created_at") == val_tweet_created_at
                ):
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

    def get_scoring_text(self, prompt: str, content: str) -> BaseRewardEvent:
        try:
            scoring_prompt = None

            scoring_prompt_text = None

            scoring_prompt = LinkContentPrompt()

            if not scoring_prompt_text:
                scoring_prompt_text = scoring_prompt.text(prompt, content)

            return scoring_prompt, [{"role": "user", "content": scoring_prompt_text}]
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

            asyncio.get_event_loop().run_until_complete(
                self.process_tweets(responses=responses)
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
                        scoring_prompt, scoring_text = self.get_scoring_text(
                            prompt, miner_tweet_text
                        )
                        scoring_messages.append({str(uid): scoring_text})
                        
            bt.logging.info(f"Executing llm_processing on {len(scoring_messages)} Link Content Relevance messages.")
            score_responses = self.reward_llm.llm_processing(scoring_messages)
            reward_events = []
            scoring_prompt = ScoringPrompt()
            for apify_score, response, uid_tensor in zip(
                scores, responses, uids
            ):  # Fixed variable name from 'response' to 'responses'
                uid = uid_tensor.item()
                reward_event = BaseRewardEvent()
                reward_event.reward = 0

                if score_responses:
                    score_result = score_responses.get(str(uid), None)
                    if score_result is None:
                        bt.logging.info(f"Link Content Relevance get_rewards: No score response for UID '{uid}'")
                        score = 0  # Default score or another logic to handle missing scores
                    else:
                        score = scoring_prompt.extract_score(score_result)
                        score /= 10.0
                    reward_event.reward = score
                if apify_score:
                    reward_event.reward = min(reward_event.reward * apify_score, 1)
                else:
                    reward_event.reward /= 2
                reward_events.append(reward_event)

                zero_scores = {}
                non_zero_scores = {}

            for (index, response), uid_tensor, reward_e in zip(enumerate(responses), uids, reward_events):
                uid = uid_tensor.item()
                if reward_e.reward == 0:
                    # score_explain = score_responses.get(str(uid), "")
                    zero_scores[uid] = reward_e.rewar
                else:
                    non_zero_scores[uid] = reward_e.reward

            bt.logging.info(f"==================================Links Content scoring Zero Scores  ({len(zero_scores)} cases)==================================")
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(f"==================================Links Content scoring Non-Zero Scores ({len(non_zero_scores)} cases)==================================")
            bt.logging.info(json.dumps(non_zero_scores))
            return reward_events
        except Exception as e:
            bt.logging.error(f"Link Content Relevance get_rewards: {str(e)}")
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events
