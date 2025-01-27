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
import bittensor as bt
import random
from typing import List, Optional
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import (
    TwitterScraperTweet,
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
)
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor
from datura.services.twitter_api_wrapper import TwitterAPIClient
from neurons.validators.reward.embedding import Embedder
from datura.services.twitter_utils import TwitterUtils
from datura.utils import (
    clean_text,
    format_text_for_match,
    is_valid_tweet,
    scrape_tweets_with_retries,
    calculate_similarity_percentage,
)
import json
from datetime import datetime
import pytz
from datetime import datetime
import pytz

APIFY_LINK_SCRAPE_AMOUNT = 3
SIMILARITY_THRESHOLD = 80.0

# Only a percentage-based threshold:
INT_DIFF_PERCENT = 0.01  # 1% difference allowed

embedder = Embedder()


class TwitterBasicSearchContentRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.twitter_content_relevance.value

    def __init__(self, device: str, scoring_type: None):
        super().__init__()
        self.device = device
        self.scoring_type = scoring_type
        self.twitter_utils = TwitterUtils()

    def clean_text(self, text):
        return clean_text(text)

    async def process_tweets(self, responses: List[TwitterSearchSynapse]):

        default_val_score_responses = [{} for _ in responses]

        try:
            start_time = time.time()
            responses_random_links = [[] for _ in responses]
            all_links = []

            # 1) Collect & sample URLs from each synapse.results
            for response, random_links in zip(responses, responses_random_links):
                tweet_urls = [
                    tweet["url"] for tweet in response.results if "url" in tweet
                ]
                if tweet_urls:
                    sample_links = random.sample(
                        tweet_urls,
                        min(APIFY_LINK_SCRAPE_AMOUNT, len(tweet_urls)),
                    )
                    all_links.extend(sample_links)
                    random_links.extend(sample_links)

            unique_links = list(set(all_links))
            if len(unique_links) == 0:
                bt.logging.info("No unique links found to process (no tweet URLs).")
                return default_val_score_responses

            bt.logging.info(f"Fetching {len(unique_links)} unique Twitter links.")
            tweets_list, non_fetched_links = await scrape_tweets_with_retries(
                unique_links, group_size=200, max_attempts=4
            )

            # 2) For each response, match tweets by ID and append to validator_tweets
            for response, random_links in zip(responses, responses_random_links):
                ids = [
                    self.twitter_utils.extract_tweet_id(link) for link in random_links
                ]
                for fetched_tweet in tweets_list:
                    if fetched_tweet.id in ids:
                        # Append the newly fetched tweet to validator_tweets
                        response.validator_tweets.append(fetched_tweet)

            end_time = time.time()
            bt.logging.info(
                f"Fetched Twitter links took {end_time - start_time:.2f}s. "
                f"All links: {len(all_links)}, Unique: {len(unique_links)}, "
                f"Fetched: {len(tweets_list)}"
            )
            bt.logging.info(
                f"Non-fetched count: {len(non_fetched_links)}, List: {non_fetched_links}"
            )

            return default_val_score_responses

        except Exception as e:
            bt.logging.error(f"Error in process_tweets: {str(e)}")
            return default_val_score_responses

    def _text_similarity(self, text1: str, text2: str) -> bool:
        # Clean each text first
        clean1 = format_text_for_match(text1)
        clean2 = format_text_for_match(text2)

        emb1 = embedder.embed_text(clean1)
        emb2 = embedder.embed_text(clean2)

        sim_percent = calculate_similarity_percentage(emb1, emb2)
        return sim_percent >= SIMILARITY_THRESHOLD

    def _int_filter_difference_checker(
        self,
        miner_value: Optional[int],
        validator_value: Optional[int],
        ratio: float = INT_DIFF_PERCENT,
    ) -> bool:
        """
        Returns True if the absolute difference between miner_value and validator_value
        is within the specified percentage threshold of the validator_value.
        Example: If ratio=0.01 (1%) and validator_value=100, allowed_diff=1 (±1).
        """
        if miner_value is None or validator_value is None:
            return False

        # Calculate the allowed difference from the validator_value (ground truth)
        allowed_diff = int(validator_value * ratio)
        return abs(miner_value - validator_value) <= allowed_diff

    def _compare_dates(self, miner_dt_str: str, validator_dt_str: str) -> bool:
        try:
            dt_validator = (
                datetime.strptime(validator_dt_str, "%a %b %d %H:%M:%S %z %Y")
                .astimezone(pytz.UTC)
                .replace(second=0, microsecond=0)
            )
            dt_miner = (
                datetime.strptime(miner_dt_str, "%a %b %d %H:%M:%S %z %Y")
                .astimezone(pytz.UTC)
                .replace(second=0, microsecond=0)
            )
            # Compare exact minute-level match in UTC
            return dt_validator == dt_miner
        except Exception as e:
            bt.logging.warning(f"Date parsing error: {str(e)}")
            return False

    def check_tweet_content(self, response: bt.Synapse) -> float:

        try:
            # 1) Gather miner & validator tweets
            miner_data_list = response.results
            validator_tweets = response.validator_tweets

            # 2) Build map of miner tweets by ID
            miner_map = {}
            for tweet_dict in miner_data_list:
                if "id" in tweet_dict:
                    miner_map[tweet_dict["id"]] = tweet_dict

            tweet_scores = []

            # 3) Iterate over validator tweets
            for val_tweet in validator_tweets:
                current_score = 0

                # Match miner tweet by ID
                if not val_tweet.id or val_tweet.id not in miner_map:
                    tweet_scores.append(0)
                    continue

                miner_tweet_data = miner_map[val_tweet.id]
                if not is_valid_tweet(miner_tweet_data):
                    tweet_scores.append(0)
                    continue

                miner_tweet = TwitterScraperTweet(**miner_tweet_data)

                # b) If it's TwitterIDSearchSynapse => confirm val_tweet.id == response.id
                if isinstance(response, TwitterIDSearchSynapse):
                    if val_tweet.id != response.id:
                        tweet_scores.append(0)
                        continue

                # c) If it's TwitterURLsSearchSynapse => confirm val_tweet.url is in response.urls
                if isinstance(response, TwitterURLsSearchSynapse):
                    if not val_tweet.url or (val_tweet.url not in response.urls):
                        tweet_scores.append(0)
                        continue

                # d) If it's TwitterSearchSynapse => check min_likes/min_retweets/min_replies
                if isinstance(response, TwitterSearchSynapse):
                    if response.min_likes is not None:
                        if (
                            val_tweet.like_count is None
                            or val_tweet.like_count < response.min_likes
                        ):
                            tweet_scores.append(0)
                            continue
                    if response.min_retweets is not None:
                        if (
                            val_tweet.retweet_count is None
                            or val_tweet.retweet_count < response.min_retweets
                        ):
                            tweet_scores.append(0)
                            continue
                    if response.min_replies is not None:
                        if (
                            val_tweet.reply_count is None
                            or val_tweet.reply_count < response.min_replies
                        ):
                            tweet_scores.append(0)
                            continue

                miner_text = (miner_tweet.text).strip()
                validator_text = (val_tweet.text).strip()
                if not self._text_similarity(miner_text, validator_text):
                    tweet_scores.append(0)
                    continue

                # Compare numeric fields with difference allowance
                if not self._int_filter_difference_checker(
                    miner_tweet.retweet_count, val_tweet.retweet_count
                ):
                    tweet_scores.append(0)
                    continue
                if not self._int_filter_difference_checker(
                    miner_tweet.reply_count, val_tweet.reply_count
                ):
                    tweet_scores.append(0)
                    continue
                if not self._int_filter_difference_checker(
                    miner_tweet.like_count, val_tweet.like_count
                ):
                    tweet_scores.append(0)
                    continue
                if not self._int_filter_difference_checker(
                    miner_tweet.quote_count, val_tweet.quote_count
                ):
                    tweet_scores.append(0)
                    continue
                if not self._int_filter_difference_checker(
                    miner_tweet.bookmark_count, val_tweet.bookmark_count
                ):
                    tweet_scores.append(0)
                    continue

                if miner_tweet.user.username.strip() != val_tweet.user.username.strip():
                    tweet_scores.append(0)
                    continue

                # Compare created_at if both exist
                if miner_tweet.created_at and val_tweet.created_at:
                    if not self._compare_dates(
                        miner_tweet.created_at, val_tweet.created_at
                    ):
                        tweet_scores.append(0)
                        continue
                elif (miner_tweet.created_at and not val_tweet.created_at) or (
                    val_tweet.created_at and not miner_tweet.created_at
                ):
                    # Mismatch if only one has a created_at
                    tweet_scores.append(0)
                    continue

                # All checks passed => score = 1
                current_score = 1
                tweet_scores.append(current_score)

            # Return average of all validated tweets
            return sum(tweet_scores) / len(tweet_scores) if tweet_scores else 0.0

        except Exception as e:
            bt.logging.error(f"check_tweet_content error: {str(e)}")
            return 0.0

    async def get_rewards(
        self, responses: List[TwitterSearchSynapse], uids: List[int]
    ) -> List[BaseRewardEvent]:

        try:
            # Step 1: fetch and fill validator_tweets
            _ = await self.process_tweets(responses=responses)

            reward_events = []
            zero_scores = {}
            non_zero_scores = {}
            grouped_val_score_responses = {}

            # Step 2: for each response, compute a final score
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score = self.check_tweet_content(response)
                bt.logging.info(f"UID {uid}: check_tweet_content => {final_score}")

                # Step 3: create a reward event
                reward_event = BaseRewardEvent()
                reward_event.reward = final_score
                reward_events.append(reward_event)

                # Keep track of final_score for logging
                if final_score == 0:
                    zero_scores[uid] = final_score
                else:
                    non_zero_scores[uid] = final_score

                # Populate grouped_val_score_responses with final_score
                grouped_val_score_responses[uid] = final_score

            # Step 4: Log zero vs. non-zero
            bt.logging.info(
                f"========== Twitter Link Content Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Twitter Link Content Non-Zero Scores ({len(non_zero_scores)} cases) ========"
            )
            bt.logging.info(json.dumps(non_zero_scores))

            return reward_events, grouped_val_score_responses

        except Exception as e:
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str))

            # On exception, return zeroed events
            reward_events = []
            for _ in responses:
                revent = BaseRewardEvent()
                revent.reward = 0
                reward_events.append(revent)

            return reward_events, {}
