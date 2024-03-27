from openai import AsyncOpenAI
from template.protocol import DiscordPromptAnalysisResult, ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As a Discord data analyst, your task is to provide users with a clear and concise summary derived from the given Discord data and the user's query.

Tasks:
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
3. You would explain how you did retrieve data based on Analysis of <UserPrompt>.

Output Guidelines (Tasks):
1. Relevant Links: Provide a selection of Discord links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
Synthesize insights from both the <UserPrompt> and the <DiscordData> to formulate a well-rounded response.
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
3. You would explain how you did retrieve data based on <PromptAnalysis>.

Operational Rules:
1. No <DiscordData> Scenario: If no DiscordData is provided, inform the user that current Discord insights related to their topic are unavailable.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided <DiscordData>" in responses. Assume user awareness of the data integration process.
4. Please separate your responses into sections for easy reading.
6. Not return text like <UserPrompt>, <PromptAnalysis>, <PromptAnalysis> to your response, make response easy to understand to any user.
7. Make headers bold using Markdown.
8. Start text with bold text "Discord Summary:".
"""


async def summarize_discord_data(
    prompt: str,
    model: str,
    filtered_messages,
    prompt_analysis: DiscordPromptAnalysisResult,
):

    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <PromptAnalysis> I analyze that prompts and generate query for API, keywords, hashtags, user_mentions.
    In <DiscordData>, Provided Discord API fetched data.

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <DiscordData>
    {filtered_messages}
    </DiscordData>

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

    return res, ScraperTextRole.DISCORD_SUMMARY


def prepare_messages_data_for_summary(messages):
    return messages.get("body", [])
