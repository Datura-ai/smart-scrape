from openai import AsyncOpenAI
from datura.dataset.tool_return import ResponseOrder
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)

SYSTEM_MESSAGE = """
As a Discord data analyst, your task is to provide users with a clear and concise answer derived from the given Discord chat conversation and the user's query.

Output Guidelines (Tasks):
1. Discord Search Summary: Analyze the user's prompt and messages, and provide easy to follow instructions or information based on the analysis. You can include any count of messages in the summary.
2. Discord Messages: Analyze DiscordData and Return list of messages, where key is DiscordData.id and value is list of possible reply IDs DiscordData.possible_replies.id. Choose the most relevant possible replies that are related to parent message the user's query.

<OutputExample>
**Discord Search Summary:**

To register Bittensor wallet address, you can visit following link: [Create Bittensor Wallet](https://docs.bittensor.com/getting-started/wallets).
Once you have registered your wallet address, you can start earning rewards by participating in the Bittensor network.

**Discord Messages:**
- { "id1": [possible_reply_id1, possible_reply_id2] }
- { "id2": [possible_reply_id1, possible_reply_id2] }
</OutputExample>


Discord Search Summary Rules:
1. No Discord Data Scenario: If no Discord data is provided, inform the user that there are no related chat messages from users.
2. Avoid explicitly stating "Based on the provided Discord data" in responses. Assume user awareness of the data integration process.
3. Structure Responses: Separate responses into sections for easy reading.
4. User-Friendly Language: Do not return text like <UserPrompt>; make responses easy to understand for any user.
5. Use Markdown: Make headers bold using Markdown, format steps, code, lists, etc.
6. Provide Links: Include links to relevant resources or information if necessary

Discord Messages Rules:
1. Amount of messages does not matter, but the quality of the messages does. In the example above you have 2 messages, but you can have more or less.
2. Include only the most relevant messages that are related to the user's query.
3. Do not include discord messages or replies with empty content.
4. Do not include discord messages with duplicate id in bullet lists.
"""


async def summarize_discord_data(
    prompt: str,
    model: str,
    filtered_messages,
    response_order: ResponseOrder
):
    content = f"""
    In <UserPrompt> provided User's prompt (Question).
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
        "id": message.get("id", ""),
        "content": message.get("content", ""),
        "channel": message.get("channel", ""),
        "author": message.get("author", ""),
    }


def prepare_messages_data_for_summary(messages):
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
