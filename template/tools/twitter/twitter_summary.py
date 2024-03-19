from openai import AsyncOpenAI
from template.protocol import TwitterPromptAnalysisResult, ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As a Twitter data analyst, your task is to provide users with a clear and concise summary derived from the given Twitter data and the user's query.

Tasks:
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
3. You would explain how you did retrieve data based on Analysis of <UserPrompt>.

Output Guidelines (Tasks):
1. Relevant Links: Provide a selection of Twitter links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
Synthesize insights from both the <UserPrompt> and the <TwitterData> to formulate a well-rounded response.
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
3. You would explain how you did retrieve data based on <PromptAnalysis>.

Operational Rules:
1. No <TwitterData> Scenario: If no TwitterData is provided, inform the user that current Twitter insights related to their topic are unavailable.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided <TwitterData>" in responses. Assume user awareness of the data integration process.
4. Please separate your responses into sections for easy reading.
5. Use <TwitterData>.url for generate tweet link, example: [username](<TwitterData>.url)
6. Not return text like <UserPrompt>, <PromptAnalysis>, <PromptAnalysis> to your response, make response easy to understand to any user.
7. Make headers bold using Markdown.
8. Start text with bold text "Twitter Summary:".
9. Return up to 10 Twitter links if available.
"""


async def summarize_twitter_data(
    prompt: str,
    model: str,
    filtered_tweets,
    prompt_analysis: TwitterPromptAnalysisResult,
):

    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <PromptAnalysis> I analyze that prompts and generate query for API, keywords, hashtags, user_mentions.
    In <TwitterData>, Provided Twitter API fetched data.
    
    <UserPrompt>
    {prompt}
    </UserPrompt>

    <TwitterData>
    {filtered_tweets}
    </TwitterData>

    <PromptAnalysis>
    {prompt_analysis}
    </PromptAnalysis>
    """

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]

    res = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    return res, ScraperTextRole.TWITTER_SUMMARY


def prepare_tweets_data_for_summary(tweets):
    data = []

    users = tweets.get("includes", {}).get("users", [])

    for tweet in tweets.get("data", []):
        author_id = tweet.get("author_id")

        author = (
            next((user for user in users if user.get("id") == author_id), None) or {}
        )

        data.append(
            {
                "id": tweet.get("id"),
                "text": tweet.get("text"),
                "author_id": tweet.get("author_id"),
                "created_at": tweet.get("created_at"),
                "url": "https://twitter.com/{}/status/{}".format(
                    author.get("username"), tweet.get("id")
                ),
                "username": author.get("username"),
            }
        )

    return data
