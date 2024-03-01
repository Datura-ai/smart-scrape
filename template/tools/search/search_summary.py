from openai import AsyncOpenAI
from template.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As Google search data analyst, your task is to provide users with a clear and concise summary derived from the given Google search data and the user's query.

Tasks:
1. Relevant Links: Provide a selection of Google search links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
Synthesize insights from both the <UserPrompt> and the <GoogleSearch> to formulate a well-rounded response.
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.

Output Guidelines (Tasks):
1. Relevant Links: Provide a selection of Google links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
Synthesize insights from both the <UserPrompt> and the <GoogleSearch> to formulate a well-rounded response.
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.

Operational Rules:
1. No <GoogleSearch> Scenario: If no GoogleSearch is provided, inform the user that current Google insights related to their topic are unavailable.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided <GoogleSearch>" in responses. Assume user awareness of the data integration process.
4. Please separate your responses into sections for easy reading.
5. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
6. Make headers bold using Markdown.
7. Start text with bold text "Search Summary:".
"""


async def summarize_search_data(prompt: str, model: str, data):
    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <GoogleSearch> I fetch data from Google search API.

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <GoogleSearch>
    {data}
    </GoogleSearch>
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

    return res, ScraperTextRole.SEARCH_SUMMARY
