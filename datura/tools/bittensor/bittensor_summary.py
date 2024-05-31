from openai import AsyncOpenAI
from datura.dataset.tool_return import ResponseOrder
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


def system_message(response_order: ResponseOrder):
    output_example = ""
    if response_order == ResponseOrder.LINKS_FIRST:
        output_example = """
            Key Sources:
                - [Installation guide for Bittensor](https://docs.bittensor.com/getting-started/installation)
                - [Apple Silicon installation guide for Bittensor](https://docs.bittensor.com/getting-started/installation#installing-on-apple-silicon)
            Bittensor Documentation Summary:
             To install Bittensor, follow the provided steps. You can install Bittensor using pip3 with the command `pip3 install bittensor --no-deps`. Alternatively, you can install it from the source by cloning the Bittensor repository from GitHub and then installing it using `python3 -m pip install -e bittensor/`. Before you start developing, ensure that you have installed Bittensor and created a Bittensor wallet.
        """
    else:
        output_example = """
            Bittensor Documentation Summary:
             To install Bittensor, follow the provided steps. You can install Bittensor using pip3 with the command `pip3 install bittensor --no-deps`. Alternatively, you can install it from the source by cloning the Bittensor repository from GitHub and then installing it using `python3 -m pip install -e bittensor/`. Before you start developing, ensure that you have installed Bittensor and created a Bittensor wallet.
            Key Sources:
                - [Installation guide for Bittensor](https://docs.bittensor.com/getting-started/installation)
                - [Apple Silicon installation guide for Bittensor](https://docs.bittensor.com/getting-started/installation#installing-on-apple-silicon)
        """

    return f"""
    As a Bittensor Documentation data analyst, your task is to provide users with a clear and concise answer derived from the given Bittensor Documentation and the user's query.

    Output Guidelines (Tasks):
    1. Analyze the user's prompt and the provided Bittensor Documentation data and write a well-rounded and detailed answer that addresses the user's query.

    <OutputExample>
    {output_example}
    </OutputExample>

    Operational Rules:
    1. No Bittensor Documentation Data Scenario: If no Bittensor documentation data is provided, inform the user that there are no related documentation.
    2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
    3. Seamless Integration: Avoid explicitly stating "Based on the provided Bittensor Documentation data" in responses. Assume user awareness of the data integration process.
    5. User-Friendly Language: Do not return text like <UserPrompt>; make responses easy to understand for any user.
    6. Use Markdown: Make headers bold using Markdown, code blocks with markdown code blocks, and lists with markdown lists.
    7. Provide Links with Unique and Descriptive Titles: Include links to relevant resources or information, and ensure that the link titles (the text that appears within the square brackets) are unique and descriptive, providing relevant information about the content or purpose of the linked resource. The link titles should be generated based on the URL itself, rather than using generic or repetitive text. For instance, if the URL is a file path within a specific repository, the link title can include the file name along with the repository name. Always include the current document's original URL link for reference.
    """


async def summarize_bittensor_data(
    prompt: str,
    model: str,
    docs,
    response_order: ResponseOrder
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
        {"role": "system", "content": system_message(response_order)},
        {"role": "user", "content": content},
    ]

    res = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    return res, ScraperTextRole.BITTENSOR_SUMMARY
