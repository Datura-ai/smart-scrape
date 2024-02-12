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
from utils.prompts import SummaryRelevancePrompt, LinkContentPrompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from template.protocol import TwitterScraperStreaming, TwitterScraperTweet
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor
from template.services.twitter_api_wrapper import TwitterAPIClient
import random
import asyncio
import re
import html
import random

def init_tokenizer(device):
    # https://huggingface.co/VMware/open-llama-7b-open-instruct
    # Fast tokenizer results in incorrect encoding, set the use_fast = False parameter.
    tokenizer = AutoTokenizer.from_pretrained(
        LinkContentRelevanceModel.reward_model_name, use_fast=False
    )
    # Generative default expects most recent token on right-hand side with padding on left.
    # https://github.com/huggingface/transformers/pull/10552
    tokenizer.padding_side = "left"

    # Check if the device is CPU or CUDA and set the precision accordingly
    torch_dtype = torch.float32 if device == 'cpu' else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        LinkContentRelevanceModel.reward_model_name, torch_dtype=torch_dtype
    ).to(device)

    return tokenizer, model

class LinkContentRelevanceModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.prompt.value

    def __init__(self, device: str, scoring_type: None, tokenizer= None, model = None, is_disable_tokenizer_reward=False):
        super().__init__()
        self.device = device

        if not is_disable_tokenizer_reward:
            if not tokenizer:
                tokenizer, model = init_tokenizer(device)
                self.tokenizer = tokenizer
                self.model = model
            else:
                self.tokenizer = tokenizer
                self.model = model
            
        self.scoring_type = scoring_type
        self.is_disable_tokenizer_reward = is_disable_tokenizer_reward
        self.tw_client = TwitterAPIClient()


    async def process_tweets(self, responses):
        try:
            start_time = time.time()
            all_links = [random.choice(response.completion_links) for response in responses if response.completion_links]
            unique_links = list(set(all_links))  # Remove duplicates to avoid redundant tasks
            tweets_list = await TwitterScraperActor().get_tweets(urls=unique_links)
            for response in responses:
                ids = [self.tw_client.extract_tweet_id(link) for link in response.completion_links]

                for tweet in tweets_list:
                    if tweet.id in ids:
                        response.validator_tweets.append(tweet)
            end_time = time.time()
            bt.logging.info(f"Fetched Twitter links method took {end_time - start_time} seconds")
        except Exception as e:
            bt.logging.error(f"Error in process_tweets: {e}")
            return
        
    def format_text_for_match(self, text):
        # url shorteners can cause problems with tweet verification, so remove urls from the text comparison.
        text = re.sub(r'(https?://)?\S+\.\S+\/?(\S+)?', '', text)
        # Some scrapers put the mentions at the front of the text, remove them.
        text = re.sub(r'^(@\w+\s*)+', '', text)
        # And some trim trailing whitespace at the end of newlines, so ignore whitespace.
        text = re.sub(r'\s+', '', text)
        # And some have special characters escaped as html entities
        text = html.unescape(text)
        # The validator apify actor uses the tweet.text field and not the note_tweet field (> 280) charts, so only
        # use the first 280 chars for comparison.
        text = text[:280]
        return text

    def check_response_random_tweet(self, response: TwitterScraperStreaming):
        try:
            tweet_score = 0
            miner_tweets = response.miner_tweets

            miner_tweets_data = miner_tweets.get('data', [])
            # miner_tweets_meta = miner_tweets.get('meta', {})
            # miner_tweets_users = miner_tweets.get('includes', {}).get('users', [])
            miner_tweets_amount = miner_tweets.get('meta', {}).get('result_count', 0)

            # Assign completion links and validator tweets from the response
            completion_links = response.completion_links
            validator_tweets = response.validator_tweets

            # Check if there are no completion links, no miner tweets, or no validator tweets
            if not completion_links or miner_tweets_amount == 0 or not response.validator_tweets:
                return 0

            # Select a random tweet from the validator's tweets
            val_tweet = random.choice(response.validator_tweets)
            
            # Extract content, ID, and creation time of the validator tweet
            val_tweet_content = val_tweet.full_text
            val_tweet_id = val_tweet.id
            val_tweet_created_at = val_tweet.created_at
            # Find the corresponding miner tweet by ID
            miner_tweet = next((tweet for tweet in miner_tweets_data if tweet['id'] == val_tweet_id), None)
            # If there is no corresponding miner tweet, append a score of 0
            if miner_tweet:
                # If a corresponding miner tweet is found, extract its creation time and text
                miner_tweet_created_at = miner_tweet['created_at']
                miner_tweet_text = miner_tweet['text']
                # Prepare texts for comparison by normalizing them
                miner_text_compared = self.format_text_for_match(miner_tweet_text)
                validator_text_compared = self.format_text_for_match(val_tweet_content)
                # Compare the normalized texts and creation times of the tweets
                if miner_text_compared == validator_text_compared and miner_tweet_created_at == val_tweet_created_at:
                    # If both match, append a score of 1
                    tweet_score = 1
                else:
                    # If there is a discrepancy, log the details and append a score of 0
                    bt.logging.info(f"Discrepancy found:")
                    bt.logging.info(f"Miner tweet - Created at: {miner_tweet_created_at}, Text: {miner_text_compared}")
                    bt.logging.info(f"Validator tweet - Created at: {val_tweet_created_at}, Text: {validator_text_compared}")
                    tweet_score = 0
            else:
                bt.logging.info("No corresponding miner tweet found for the validator tweet.")
                tweet_score = 0
            return tweet_score
        except Exception as e:
            bt.logging.error(f"check_response_random_tweet: {e}")
            return 0

    def reward(self, prompt: str, content: str, response: TwitterScraperStreaming) -> BaseRewardEvent:
        try:
            reward_event = BaseRewardEvent()

            with torch.no_grad():
                # Choose correct scoring prompt for request type.
                # Determine the scoring prompt based on the provided name or the default scoring type.
                scoring_prompt = LinkContentPrompt()
                 # Format scoring prompt for this completion.
                scoring_prompt_text = scoring_prompt.text(prompt, content)
                
                if self.is_disable_tokenizer_reward:
                    length = len(response.completion_links) * 2 
                    score = length if length < 10 else 9
                    # Scale 0-10 score to 0-1 range.
                    score /= 10.0
                    reward_event.reward = score
                    return reward_event

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
                generated_text = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                # Extract score from generated text.
                score_text = generated_text[0][len(scoring_prompt_text) :]
                score = scoring_prompt.extract_score(score_text)
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

            asyncio.get_event_loop().run_until_complete(self.process_tweets(responses=responses))
            scores = [self.check_response_random_tweet(response) for response in responses]

            reward_events = []
            for score, response, uid_tensor  in zip(scores, responses, uids):  # Fixed variable name from 'response' to 'responses'
                uid = uid_tensor.item()
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                if score:
                    bt.logging.info(f"Processing score for response with miner tweets.")
                    miner_tweets = response.miner_tweets
                    miner_tweets_data = miner_tweets.get('data', [])
                    links_scores = []
                    for link in response.completion_links:
                        tweet_id = self.tw_client.extract_tweet_id(link)
                        miner_tweet = next((tweet for tweet in miner_tweets_data if tweet['id'] == tweet_id), None)
                        if miner_tweet:
                            miner_tweet_text = miner_tweet['text']
                            reward = self.reward(prompt, miner_tweet_text, response)
                            links_scores.append(reward)
                            bt.logging.info(f"UID:{uid}, Tweet ID {tweet_id} yielded a reward of {reward.reward}.")
                        else:
                            bt.logging.warning(f"UID:{uid}, No matching tweet found for ID {tweet_id}.")
                    if links_scores:
                        average_score = sum(link.reward for link in links_scores) / len(links_scores)
                        reward_event.reward = average_score
                        bt.logging.info(f"UID:{uid}, Average score calculated: {average_score}, links_scores: {[link.reward for link in links_scores]}")
                    else:
                        bt.logging.warning("UID:{uid}No link scores to average, reward remains 0.")
                reward_events.append(reward_event)

            # Iterate over responses and assign rewards based on scores
            bt.logging.info(f"==================================Links Content scoring Explanation Begins==================================")
            bt.logging.info(f"Prompt: {prompt}")
            for (index, response), uid_tensor, reward_e in zip(enumerate(responses), uids, reward_events):
                uid = uid_tensor.item()
                bt.logging.info(f"UID: {uid} | Score: {reward_e.reward:.2f}")
                bt.logging.info(f"UID:{uid} Compeltion: {response.completion}")
                bt.logging.info(f"----------------------------------------------------------------------")
            bt.logging.info(f"==================================Summary Relevance Scoring Explanation Ends==================================")
            return reward_events
        except Exception as e:
            bt.logging.error(f"Reward model issue: {e}")
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events

