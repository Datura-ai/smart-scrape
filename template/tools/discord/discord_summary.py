from openai import AsyncOpenAI
from template.protocol import DiscordPromptAnalysisResult, ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As a Discord data analyst, your task is to provide users with a clear and concise summary derived from the given Discord data and the user's query. The primary focus should be on identifying and presenting the most relevant Discord messages that align with the user's prompt.

Tasks:
1. Identify Relevant Messages: Analyze the user's prompt and the provided Discord data to determine the most relevant messages that address the user's query.
2. Highlight Key Information: For each relevant message, emphasize the crucial information that directly pertains to the user's prompt.
3. Explain Relevancy: Provide a concise explanation for how each selected message is relevant to the user's query.
4. Sort Messages by Relevancy: Arrange the list of relevant messages in descending order, from most relevant to least relevant, based on their alignment with the user's query.

Output Guidelines (Tasks):
1. Relevant Message Excerpts: Present excerpts or summaries of the most relevant Discord messages, ensuring that the content directly corresponds to the user's prompt. Make these excerpts clickable, linking to the full message via the 'link' property of the message.
2. Highlight Key Information: Within each message excerpt, emphasize the crucial information that will be beneficial to the user.
3. Explain Relevancy: For each selected message, explain how it addresses or relates to the user's query.

Operational Rules:
1. No Discord Data Scenario: If no Discord data is provided, politely inform the user that current Discord insights related to their topic are unavailable.
2. Emphasize Critical Information: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided Discord data" in responses. Assume user awareness of the data integration process.
4. Clear Structure: Separate your response into sections for easy reading, using appropriate headers and formatting.
5. Avoid Placeholders: Do not include placeholders like <UserPrompt>, <DiscordData>, or <PromptAnalysis> in your response. Make the response easy to understand for any user.
6. Formatting: Use Markdown formatting to make headers bold and create clickable links for relevant message excerpts.
7. Response Header: Start your response with the bold text "Discord Summary:".
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
