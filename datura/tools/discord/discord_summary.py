from openai import AsyncOpenAI
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)

SYSTEM_MESSAGE = """
As a Discord data analyst, your task is to provide users with a clear and concise summary derived from the given Discord data (including replies) and the user's query. The primary focus should be on identifying and presenting the most relevant Discord messages and their replies that align with the user's prompt.

Output Guidelines (Tasks):
1. Identify Relevant Messages and Replies: Analyze the user's prompt and the provided Discord data (including replies) to determine the most relevant messages and replies that address the user's query.
2. Key Discord Messages and Replies: Provide a selection of up to 20(if available) relevant messages with their links and relevant replies that directly correspond to the user's prompt. Emphasize the crucial information that directly pertains to the user's prompt for each message and reply.
3. Sort Messages by Relevancy: Arrange the list of relevant messages and replies in descending order, from most relevant to least relevant, based on their alignment with the user's query.
4. Discord Search Summary: Synthesize insights from both the user's prompt and the Discord data (including replies) to formulate a well-rounded response.

<OutputExample>
**Discord Messages:**
- Message: [Message Content and explanation](https://discord.com/channels/2/43)
- - First Reply: [Reply Content and explanation](https://discord.com/channels/2/43/456)
- - Second Reply: [Reply Content and explanation](https://discord.com/channels/2/43/789)
- Message: [Message Content and explanation](https://discord.com/channels/31/21)

**Discord Search Summary:**
Georgia, as a country, hosts a diverse range of sports events catering to various interests. Popular sports in Georgia include football, basketball, rugby union, wrestling, judo, and weightlifting. The sports industry in Georgia is thriving, with a growing interest in modern sports like rugby union, weightlifting, basketball, judo, and football. The country offers a wide array of sporting activities from traditional sports like polo to modern events like football matches, showcasing a rich sporting culture.
</OutputExample>

Operational Rules:
1. No Discord Data Scenario: If no Discord data is provided, inform the user that current insights related to their topic are unavailable.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided Discord data" in responses. Assume user awareness of the data integration process.
4. Structure Responses: Separate responses into sections for easy reading.
5. User-Friendly Language: Do not return text like <UserPrompt>; make responses easy to understand for any user.
6. Use Markdown: Make headers bold using Markdown.
7. Provide Links: Return up to 20 messages with their links if available.
8. Formatting: Do not number the "Discord Messages". Instead, provide each on a new line and their replies as numbered on the next line, spaced to the right to make it look like a sub-element of a list.
9. Maintain Order: Always maintain the order as shown in <OutputExample>, first providing "Discord Messages" (with replies), followed by "Discord Search Summary".
10. Include Explanations: For each message content and reply, include an explanation that connects its relevance to the user's question. The link's description should be 10-25 words, emphasizing the main topic from that link.
"""


async def summarize_discord_data(
    prompt: str,
    model: str,
    filtered_messages,
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


def prepare_message(message):
    return {
        "content": message.get("content", ""),
        "channel": message.get("channel", ""),
        "author": message.get("author", ""),
        "link": message.get("link", ""),
    }


def prepare_messages_data_for_summary(messages):
    messages = messages.get("body", [])

    normalized_messages = []

    for message in messages:
        normalized_messages.append(
            {
                **prepare_message(message),
                "replies": [
                    prepare_message(reply) for reply in message.get("replies", [])
                ],
                "possible_replies": [
                    prepare_message(reply)
                    for reply in message.get("possible_replies", [])
                ],
            }
        )

    return normalized_messages
