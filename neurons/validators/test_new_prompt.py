import asyncio
import torch
import bittensor as bt

from neurons.validators.reward.config import RewardScoringType
from datura.protocol import ScraperStreamingSynapse, ScraperTextRole, ResultType, Model
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.reward.summary_relevance import SummaryRelevanceRewardModel

def log_synapse_details(synapse):
    """Log all details of the given synapse for debugging purposes."""
    bt.logging.info("[CHECKER] Logging full details of the synapse...")
    print("\n=== SYNAPSE DETAILS ===")
    print(f"Prompt: {synapse.prompt}")
    print(f"Result Type: {synapse.result_type}")
    print(f"Completion: {synapse.completion}")
    print(f"Tools: {synapse.tools}")
    print(f"Model: {synapse.model}")
    print("Text Chunks:")
    for role, chunks in synapse.text_chunks.items():
        print(f"  Role {role}: {chunks}")
    print("========================\n")
    bt.logging.info("[CHECKER] Synapse details logged successfully.")

async def test_summary_relevance_model():

    # Initialize LLM and Relevance Model
    bt.logging.info("[INIT] Initializing RewardLLM and SummaryRelevanceRewardModel...")
    llm = RewardLLM()
    relevance = SummaryRelevanceRewardModel(
        device="cpu",
        scoring_type=RewardScoringType.summary_relevance_score_template,
        llm_reward=llm,
    )
    bt.logging.info("[INIT] Initialization complete.")

    # 2) Sample completion text
    bt.logging.info("[DATA] Preparing sample completion text...")
    completion_text = """\n\n\n**Key Posts:**\n- [Football - The People's Sport](https://www.reddit.com/r/football/)\n  - This link provides a platform for discussing the latest football news, match results, transfer rumors, and other football-related topics.\n"""

    # 3) Construct the synapse
    bt.logging.info("[DATA] Constructing ScraperStreamingSynapse...")
    synapse = ScraperStreamingSynapse(
        prompt="What are the recent sports news?",
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        text_chunks={ScraperTextRole.REDDIT_SUMMARY: [completion_text]},
        completion=completion_text,
        tools=["Reddit Search"],
        model=Model.NOVA 
    )
    log_synapse_details(synapse)  # Log synapse details

    # 4) Evaluate the single synapse
    bt.logging.info("[EVALUATION] Calling get_rewards on SummaryRelevanceRewardModel...")
    responses = [synapse]
    uids = torch.tensor([18])

    # Add detailed logging for get_rewards
    try:
        bt.logging.info(f"[EVALUATION] Responses: {responses}")
        bt.logging.info(f"[EVALUATION] UIDs: {uids}")
        
        reward_events, link_description_scores_list = await relevance.get_rewards(
            responses=responses, 
            uids=uids
        )
        bt.logging.info(f"[EVALUATION] Reward Events: {reward_events}")
        bt.logging.info(f"[EVALUATION] Link Description Scores: {link_description_scores_list}")
        
    except Exception as e:
        bt.logging.error(f"[ERROR] Failed during get_rewards execution: {e}")
        raise

    # 5) Print results
    bt.logging.info("[RESULTS] Printing results...")
    print("\n=== SUMMARY RELEVANCE TEST RESULTS ===")
    for idx, event in enumerate(reward_events):
        print(f"Response #{idx} => reward: {event.reward:.4f}")
    print("Link Description Scores:", link_description_scores_list)
    print("======================================")
    bt.logging.info("[RESULTS] Test completed successfully.")

if __name__ == "__main__":
    asyncio.run(test_summary_relevance_model())
