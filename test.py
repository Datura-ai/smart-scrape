import asyncio
from openai import OpenAI, AsyncOpenAI
import time
from neurons.validators.utils.prompts import ScoringPrompt
import asyncio

from neurons.validators.reward.twitter_content_relevance import (
    TwitterContentRelevanceModel,
)
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.reward import (
    RewardModelType,
    RewardScoringType,
    BaseRewardModel,
)
from neurons.validators.reward.reward import BaseRewardEvent
from datura.protocol import ScraperStreamingSynapse, ScraperTextRole
from datura.services.twitter_utils import TwitterUtils
import torch
import bittensor as bt
import json
from neurons.validators.reward.reward_llm import RewardLLM, ScoringSource


client = AsyncOpenAI(timeout=60)


prompt = "Prospects for awareness on mental health"
tweet_text = """I’m going to be on Capitol Hill advocating for mental health and it just dawned on me that all of the people I’ve been fussing at and talking about on Twitter for the past four years are the literal government 😆 that I’m about to go see. 

This is going to be a TIME!!!!!"""
tweet_id = 1795590843199561996


async def test_tweet_score():
    llm = RewardLLM()
    # llm.init_pipe_zephyr()

    relevance = TwitterContentRelevanceModel(
        device="cuda",
        scoring_type=RewardScoringType.link_content_relevance_template,
        llm_reward=llm,
    )

    result = relevance.get_scoring_text(
        prompt,
        tweet_text,
        None,
    )

    scoring_messages = []

    if result:
        scoring_prompt, scoring_text = result
        scoring_messages.append({str(tweet_id): scoring_text})

    return await get_scores(scoring_messages, ScoringSource.OpenAI)


async def get_scores(scoring_messages, source):

    reward_llm = RewardLLM()

    current_score_responses = reward_llm.get_score_by_source(
        messages=scoring_messages, source=source
    )

    # New code to extract scores from current_score_responses
    extracted_scores = {}
    scoring_prompt = ScoringPrompt()
    for tweet_id, score_result in current_score_responses.items():
        # Use the scoring_prompt.extract_score method to extract scores from score_result
        extracted_score = scoring_prompt.extract_score(
            score_result
        )  # Assuming score_result is the correct input for extract_score
        extracted_scores[tweet_id] = extracted_score
    return extracted_scores


async def main():
    st = time.time()

    tasks = [test_tweet_score() for _ in range(70)]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(result)
        # print(result.choices[0].message.content)

    end = time.time()

    print(f"Time taken: {end-st} seconds")


asyncio.run(main())