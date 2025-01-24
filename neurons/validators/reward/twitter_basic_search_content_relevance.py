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
import asyncio
import re
import html
import torch
import random
from typing import List, Optional
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import TwitterScraperTweet, TwitterSearchSynapse, Embedder
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor
from datura.services.twitter_api_wrapper import TwitterAPIClient
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
INT_DIFF_ALLOWED = 3


class TwitterBasicSearchContentRelevanceModel(BaseRewardModel):

    @property
    def name(self) -> str:
        return RewardModelType.twitter_content_relevance.value

    def __init__(self, device: str, scoring_type: None):
        super().__init__()
        self.device = device

        self.scoring_type = scoring_type
        self.tw_client = TwitterAPIClient()
        self.embedder = Embedder()

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
                    self.tw_client.utils.extract_tweet_id(link) for link in random_links
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

        emb1 = self.embedder.embed_text(clean1)
        emb2 = self.embedder.embed_text(clean2)

        sim_percent = calculate_similarity_percentage(emb1, emb2)
        return sim_percent >= SIMILARITY_THRESHOLD

    def _int_filter_difference_checker(
        self, a: Optional[int], b: Optional[int], diff_allowed: int
    ) -> bool:
        # Both not None => check difference
        return a - b <= diff_allowed

    def _compare_dates(self, miner_dt_str: str, validator_dt_str: str) -> bool:

        try:
            dt_validator = (
                datetime.strptime(validator_dt_str, "%a %b %d %H:%M:%S %z %Y")
                .astimezone(pytz.UTC)
                .replace(second=0, microsecond=0)
            )
            # TODO use same format as validator
            dt_miner = datetime.fromisoformat(
                miner_dt_str.replace("Z", "+00:00")
            ).replace(second=0, microsecond=0)
            return dt_validator == dt_miner
        except Exception as e:
            bt.logging.warning(f"Date parsing error: {str(e)}")
            return False

    def check_tweet_content(self, response: TwitterSearchSynapse) -> float:

        try:
            tweet_scores = []
            # Build a map of miner tweets by ID
            miner_map = {m["id"]: m for m in response.results if "id" in m}

            for val_tweet in response.validator_tweets:
                current_score = 0

                # 2) Find corresponding miner tweet
                if not val_tweet.id or val_tweet.id not in miner_map:
                    tweet_scores.append(0)
                    continue

                miner_tweet = miner_map[val_tweet.id]

                if not is_valid_tweet(miner_tweet):
                    tweet_scores.append(0)
                    continue

                miner_tweet = TwitterScraperTweet(**miner_tweet)

                # 3) Compare tweet text with embeddings and TwitterSearchSynapse and actual tweet data
                miner_text = (miner_tweet.text or "").strip()
                validator_text = (val_tweet.text or "").strip()
                if not self._text_similarity(miner_text, validator_text):
                    tweet_scores.append(0)
                    continue

                if (
                    response.min_likes
                    and val_tweet.like_count + INT_DIFF_ALLOWED <= response.min_likes
                ):
                    tweet_scores.append(0)
                    continue

                #    Similarly for retweets
                if (
                    response.min_retweets
                    and val_tweet.retweet_count + INT_DIFF_ALLOWED
                    <= response.min_retweets
                ):
                    tweet_scores.append(0)
                    continue

                #    Similarly for replies
                if (
                    response.min_replies
                    and val_tweet.reply_count + INT_DIFF_ALLOWED <= response.min_replies
                ):
                    tweet_scores.append(0)
                    continue

                # 4) Compare min_retweets, min_replies, min_likes
                if not self._int_filter_difference_checker(
                    miner_tweet.retweet_count, val_tweet.retweet_count, INT_DIFF_ALLOWED
                ):
                    tweet_scores.append(0)
                    continue

                if not self._int_filter_difference_checker(
                    miner_tweet.reply_count, val_tweet.reply_count, INT_DIFF_ALLOWED
                ):
                    tweet_scores.append(0)
                    continue

                if not self._int_filter_difference_checker(
                    miner_tweet.like_count, val_tweet.like_count, INT_DIFF_ALLOWED
                ):
                    tweet_scores.append(0)
                    continue

                if not self._int_filter_difference_checker(
                    miner_tweet.view_count, val_tweet.view_count, INT_DIFF_ALLOWED
                ):
                    tweet_scores.append(0)
                    continue

                if not self._int_filter_difference_checker(
                    miner_tweet.quote_count, val_tweet.quote_count, INT_DIFF_ALLOWED
                ):
                    tweet_scores.append(0)
                    continue

                # if not self._int_filter_difference_checker(
                #     miner_tweet.impression_count,
                #     val_tweet.impression_count,
                #     INT_DIFF_ALLOWED,
                # ):
                #     tweet_scores.append(0)
                #     continue

                if not self._int_filter_difference_checker(
                    miner_tweet.bookmark_count,
                    val_tweet.bookmark_count,
                    INT_DIFF_ALLOWED,
                ):
                    tweet_scores.append(0)
                    continue

                # 5) Compare username with embeddings if both exist
                # TODO compare using string
                miner_username = miner_tweet.user.username.strip()
                validator_username = val_tweet.user.username.strip()
                if miner_username or validator_username:
                    if not self._text_similarity(miner_username, validator_username):
                        tweet_scores.append(0)
                        continue

                # TODO compare id and url also

                # 6) Compare created_at if both exist
                if miner_tweet.created_at and val_tweet.created_at:
                    if not self._compare_dates(
                        miner_tweet.created_at, val_tweet.created_at
                    ):
                        tweet_scores.append(0)
                        continue
                elif (miner_tweet.created_at and not val_tweet.created_at) or (
                    val_tweet.created_at and not miner_tweet.created_at
                ):
                    # mismatch if only one has created_at
                    tweet_scores.append(0)
                    continue

                # If we reach here => all checks passed
                current_score = 1
                tweet_scores.append(current_score)

            # 8) Return average
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

            # Step 4: Log zero vs non-zero
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
