from openai import AsyncOpenAI
from datura.dataset.tool_return import ResponseOrder
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


def system_message(response_order: ResponseOrder):
    output_example = ""
    if response_order == ResponseOrder.LINKS_FIRST:
        output_example = """
            Key Sources:
                - [tool_manager.py file in smart-scrape repository](https://github.com/Datura-ai/smart-scrape/blob/main/datura/tools/tool_manager.py)
            Subnets Source Code Summary:
             To manage and execute tools in Datura, the system uses a structured process with the following key components:
             The **ToolManager Class** is the central component for managing tools. It initializes with parameters like prompt and tool names, uses `get_all_tools()` to retrieve available tools, identifies tools to use based on the prompt or manual input, and executes tools asynchronously.
             The **BaseTool and BaseToolkit Classes** define the structure for individual tools and group tools into toolkits, respectively, providing necessary summaries and event handling.
             The **Tool Execution Flow** detects and categorizes tools into toolkit actions and independent tools, executes them concurrently, and streams responses as HTTP responses.
        """
    else:
        output_example = """
            Subnets Source Code Summary:
             To manage and execute tools in Datura, the system uses a structured process with the following key components:
             The **ToolManager Class** is the central component for managing tools. It initializes with parameters like prompt and tool names, uses `get_all_tools()` to retrieve available tools, identifies tools to use based on the prompt or manual input, and executes tools asynchronously.
             The **BaseTool and BaseToolkit Classes** define the structure for individual tools and group tools into toolkits, respectively, providing necessary summaries and event handling.
             The **Tool Execution Flow** detects and categorizes tools into toolkit actions and independent tools, executes them concurrently, and streams responses as HTTP responses.
            Key Sources:
                - [tool_manager.py file in smart-scrape repository](https://github.com/Datura-ai/smart-scrape/blob/main/datura/tools/tool_manager.py)
        """

    return f"""
    As a Subnets Source Code data analyst, your task is to provide users with a clear and concise answer derived from the given Source Code and the user's query.

    Output Guidelines (Tasks):
    1. Analyze the user's prompt and the provided subnet source code data and write a well-rounded and detailed answer that addresses the user's query.

    <OutputExample>
    {output_example}
    </OutputExample>

    Operational Rules:
    1. No Source Code Data Scenario: If no source code data is provided, inform the user that there are no related source code.
    2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
    3. Seamless Integration: Avoid explicitly stating "Based on the provided subnet source code data" in responses. Assume user awareness of the data integration process.
    5. User-Friendly Language: Do not return text like <UserPrompt>; make responses easy to understand for any user.
    6. Use Markdown: Make headers bold using Markdown, code blocks with markdown code blocks, and lists with markdown lists.
    7. Provide Links with Unique and Descriptive Titles: Include links to relevant resources or information, and ensure that the link titles (the text that appears within the square brackets) are unique and descriptive, providing relevant information about the content or purpose of the linked resource. The link titles should be generated based on the URL itself, rather than using generic or repetitive text. For instance, if the URL is a file path within a specific repository, the link title can include the file name along with the repository name. Always include the current document's original URL link for reference.
    """


async def summarize_subnet_source_code_data(
    prompt: str,
    model: str,
    docs,
    response_order,
):
    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <SourceCodeData>, provided subnets source code (Data)

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <SourceCodeData>
    {docs}
    </SourceCodeData>
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

    return res, ScraperTextRole.SUBNETS_SOURCE_CODE_SUMMARY
