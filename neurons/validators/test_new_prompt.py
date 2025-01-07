import asyncio
import torch
import bittensor as bt

from neurons.validators.reward.config import RewardScoringType
from datura.protocol import ScraperStreamingSynapse, ScraperTextRole, ResultType, Model
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.reward.summary_relevance import SummaryRelevanceRewardModel

async def test_summary_relevance_model():

    # 1) Initialize the SummaryRelevanceRewardModel
    summary_model = SummaryRelevanceRewardModel(
        device="cpu",
        scoring_type=RewardScoringType.summary_relevance_score_template,
        llm_reward=RewardLLM  # Pass the class or an instance if your pipeline expects it
    )

    # 2) Sample completion text
    completion_text = """\n\n\n**Key Posts:**\n- [Football - The People's Sport](https://www.reddit.com/r/football/)\n  - This link provides a platform for discussing the latest football news, match results, transfer rumors, and other football-related topics.\n  \n- [r/nfl - National Football League Discussion](https://www.reddit.com/r/nfl/)\n  - A subreddit dedicated to discussing all things related to the National Football League (NFL).\n\n- [WNBA News and Information](https://www.reddit.com/r/wnba/)\n  - This subreddit offers a space for constructive discussions about the WNBA and professional women's basketball, covering game analysis, player performances, and trades.\n\n- [Commanders down Falcons in overtime to book return to playoffs](https://www.reddit.com/r/sports/comments/1hpf9d7/commanders_down_falcons_in_overtime_to_book/)\n  - Highlights the recent football match where the Commanders defeated the Falcons in overtime to secure a spot in the playoffs.\n\n- [Australia v India 2024-25 | Fourth Test | Day Five : r/sports](https://www.reddit.com/r/sports/comments/1hpi5uz/australia_v_india_202425_fourth_test_day_five/)\n  - Showcases an epic cricket match between Australia and India that captivated fans worldwide.\n\n**Reddit Summary:**\nThe recent sports news covers a wide range of topics, including football, NFL, WNBA, and cricket. From discussions on football match results and transfer rumors to the excitement of NFL playoffs and epic cricket contests between powerhouse nations like Australia and India, the sports world is abuzz with thrilling events and updates. Fans can engage in constructive discussions, analysis of player performances, and stay updated on the latest developments across various sports leagues and tournaments.\n\n\n**Key News:**\n- [Chess grandmaster Magnus Carlsen rejoins tournament](https://news.ycombinator.com/item?id=42549226)\n- [Carlsen quits World Rapid and Blitz championship after tensions with FIDE](https://news.ycombinator.com/item?id=42527572)\n- [How can NBA address 3-point boom? Ranking 12 potential solutions](https://news.ycombinator.com/item?id=42534341)\n- [Performance of LLMs on Advent of Code 2024](https://news.ycombinator.com/item?id=42551863)\n\n**Hacker News Summary:**\nRecent sport news highlights the involvement of chess grandmaster Magnus Carlsen in tournaments, including his decision to rejoin and quit championships due to tensions with FIDE. Additionally, discussions on the NBA addressing the 3-point boom and the performance of LLMs on Advent of Code 2024 have been significant topics of interest in the sports community. These events shed light on the competitive nature and challenges faced in the world of sports.\n\n\n**Key Sources:**\n- [Latest Sports News from ESPN.](http://www.espn.com/espn/latestnews) NBA highlights, NFL updates, and more.\n- [CNN Sports Section.](https://www.cnn.com/sport) Global sports news with in-depth analysis and videos.\n- [US News Sports Section.](https://www.usnews.com/news/sports) Updates on Roki Sasaki and Japanese free agency.\n- [NBC Sports News.](https://www.nbcnews.com/sports) Washington's NFL postseason trip and more.\n- [AP News Sports Section.](https://apnews.com/sports) Man United's loss, Verona's win, and more.\n- [Reuters Sports Section.](https://www.reuters.com/sports/) Cricket updates and Premier League highlights.\n- [New York Times Sports Section.](https://www.nytimes.com/section/sports) Vikings vs. Lions matchup and LeBron James' career.\n- [USA Today Sports Section.](https://www.usatoday.com/sports/) Tua Tagovailoa's status and Aaron Rodgers' benching.\n- [ESPN Sports Coverage.](https://www.espn.com/) Top headlines and sports fan updates.\n- [NBC Sports NFL News.](https://www.nbcsports.com/nfl/profootballtalk/rumor-mill/news/monday-night-football-kerby-joseph-jared-goff-carry-lions-to-40-34-win-over-49ers) Monday Night Football highlights and NFL mock drafts.\n\n**Search Summary:**\nStay updated with the latest sports news from various sources covering NBA, NFL, cricket, Premier League, and more. Get insights on player injuries, game highlights, and upcoming events in the sports world.\n\n**Summary:**\nRecent sports news covers a diverse range of topics including football, NFL, WNBA, cricket, chess, and NBA. The sports world witnessed the Commanders defeating the Falcons in overtime to secure a playoff spot, epic cricket matches between Australia and India, and discussions on NBA addressing the 3-point boom. Chess grandmaster Magnus Carlsen's tournament involvement, including rejoining and quitting championships due to tensions with FIDE, has also been a significant highlight. Stay informed about player performances, game results, and upcoming events across various sports leagues and tournaments through different news sources.
"""

    # 3) Construct the synapse using the required fields in the constructor
    synapse = ScraperStreamingSynapse(
        prompt="What are the recent sports news?",
        result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        # We'll store the entire 'completion_text' in SEARCH_SUMMARY for demonstration:
        text_chunks={
            ScraperTextRole.SEARCH_SUMMARY: [completion_text]
        },
        completion=completion_text,
        model=Model.NOVA 
    )

    # 4) Evaluate the single synapse
    responses = [synapse]
    uids = torch.tensor([18])

    reward_events, link_description_scores_list = await summary_model.get_rewards(
        responses=responses, 
        uids=uids
    )

    # 5) Print results
    print("\n=== SUMMARY RELEVANCE TEST RESULTS ===")
    for idx, event in enumerate(reward_events):
        print(f"Response #{idx} => reward: {event.reward:.4f}")
    print("Link Description Scores:", link_description_scores_list)
    print("======================================")

if __name__ == "__main__":
    asyncio.run(test_summary_relevance_model())
