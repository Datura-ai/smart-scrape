from openai import AsyncOpenAI
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As a Bittensor docs analyst, your task is to provide users with a clear and concise summary derived from the given Bittensor documentation.

Output Guidelines (Tasks):
1. Key links: Provide a selection of links to specific Bittensor documentation parts that directly correspond to the <UserPrompt>.
Synthesize insights from both the <UserPrompt> and the <BittensorData> to formulate a well-rounded response.
2. Summarizes key posts

<OutputExample>
Bittensor Key Links:
    - [Learn Bittensor Concepts](https://docs.bittensor.com/learn/introduction)
    - [Bittensor Building Blocks](https://docs.bittensor.com/learn/bittensor-building-blocks)
Bittensor Summary:
    Bittensor is a decentralized machine learning platform that enables developers to build, train, and deploy machine learning models on a decentralized network. The platform provides a set of tools and libraries that allow developers to create and deploy machine learning models on the Bittensor network.
</OutputExample>

Operational Rules:
1. No <BittensorData> Scenario: If no BittensorData is provided, inform the user that current Documentation insights related to their topic are unavailable.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided <BittensorData>" in responses. Assume user awareness of the data integration process.
4. Please separate your responses into sections for easy reading.
5. For each link title, include a concise explanation that connects its relevance to the user's question. Use <BittensorData>.url for link
6. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
7. Make headers bold using Markdown.
8. Return up to 10 bittensor documentation links if available.
9. Do not number the "key posts"; instead, provide each on a new line.
10. Always maintain the order as shown in <OutputExample>, first providing "Key Posts", followed by "Bittensor Documentation Summary".
"""


async def summarize_bittensor_data(
    prompt: str,
    model: str,
    docs,
):
    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <BittensorData>, Provided Bittensor API fetched data.

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <BittensorData>
    {docs}
    </BittensorData>
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

    return res, ScraperTextRole.BITTENSOR_SUMMARY
