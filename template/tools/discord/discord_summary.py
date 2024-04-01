from openai import AsyncOpenAI
from template.protocol import DiscordPromptAnalysisResult, ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As a Discord data analyst, your task is to provide users with a clear and concise summary derived from the given Discord data and the user's query.
The primary focus should be on identifying and presenting the most relevant Discord messages that align with the user's prompt.

Output Guidelines (Tasks):
1. Identify Relevant Messages: Analyze the user's prompt and the provided Discord data to determine the most relevant messages that address the user's query.
2. Key Links: Provide a selection of links that directly correspond to the <UserPrompt>.
Synthesize insights from both the <UserPrompt> and the <DiscordData> to formulate a well-rounded response.
3. Highlight Key Information: For each relevant message, emphasize the crucial information that directly pertains to the user's prompt.
4. Sort Messages by Relevancy: Arrange the list of relevant messages in descending order, from most relevant to least relevant, based on their alignment with the user's query.

<OutputExample>
Key Sources:
    [Title and explanation.](https://discord.com/channels/799672011265015819/1161764867166961704/1210526845196439562)
    [Title and explanation.](https://discord.com/channels/799672011265015819/1161764867166961704/1210526845196439562)
Search Summary:
 Georgia, as a country, hosts a diverse range of sports events catering to various interests. Popular sports in Georgia include football, basketball, rugby union, wrestling, judo, and weightlifting. The sports industry in Georgia is thriving, with a growing interest in modern sports like rugby union, weightlifting, basketball, judo, and football. The country offers a wide array of sporting activities from traditional sports like polo to modern events like football matches, showcasing a rich sporting culture.
</OutputExample>

Operational Rules:
1. No <DiscordData> Scenario: If no DiscordData is provided, inform the user that current insights related to their topic are unavailable.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided <DiscordData>" in responses. Assume user awareness of the data integration process.
4. Please separate your responses into sections for easy reading.
5. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
6. Make headers bold using Markdown.
8. Return up to 10 links if available.
9. Do not number the "key Sources"; instead, provide each on a new line.
10. lways maintain the order as shown in <OutputExample>, first providing "Key Sources", followed by "DiscordSearch Summary".
11. For each link, include a explanation that connects its relevance to the user's question. The link's description should be 10-25 words, which emphasizes the main topic from that link. [Title and explanation.](https://discord.com/channels/799672011265015819/1161764867166961704/1210526845196439562)
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
