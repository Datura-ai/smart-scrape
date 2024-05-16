from openai import AsyncOpenAI
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As a Subnets Source Code data analyst, your task is to provide users with a clear and concise answer derived from the given Source Code and the user's query.

Output Guidelines (Tasks):
1. Analyze the user's prompt and the provided subnet source code data and write a well-rounded and detailed answer that addresses the user's query.

<OutputExample>
**Subnets Source Code Summary:**

The tools system in Datura is implemented through a structured process that involves the use of toolkits, tools, and tool managers. Here is an overview of how the tools system is implemented in Datura based on the provided code snippets:

1. **ToolManager Class**:
   - The `ToolManager` class serves as a central component for managing tools in Datura. It initializes with various parameters such as prompt, manual tool names, send function, language, region, and date filter.
   - It interacts with the `ResponseStreamer` to handle streaming responses and sending events.
   - The `ToolManager` class utilizes the `get_all_tools()` function to retrieve a list of all available tools and creates a mapping of tool names to tool instances.
   - It includes a method `detect_tools_to_use()` to identify which tools to use based on the provided prompt. If manual tool names are provided, it uses them; otherwise, it identifies tools based on the prompt using a predefined template.
   - The `run()` method orchestrates the execution of detected tools by categorizing them into toolkit actions and independent tools, then running them asynchronously.

2. **BaseTool and BaseToolkit Classes**:
   - The `BaseTool` class defines the structure for individual tools, including attributes like `tool_id`, `slug`, and `tool_manager`. It also includes an abstract method `send_event()` for sending events related to the tool.
   - The `BaseToolkit` class represents a collection of tools grouped under a specific toolkit. It contains information such as `toolkit_id`, `name`, `description`, and `slug`. It includes abstract methods `get_tools()` to retrieve tools within the toolkit and `summarize()` to provide a summary based on the prompt and data.

3. **Tool Execution Flow**:
   - The tool execution flow involves detecting tools to use, categorizing them into toolkit actions and independent tools, and running them concurrently using asyncio tasks.
   - Tools belonging to a toolkit are executed together, while independent tools are executed separately.
   - The response from the tools is streamed and compiled to form a completion response body, which is then sent as an HTTP response.

For more detailed information, you can refer to the [Tool Manager Python File](https://github.com/surcyf123/smart-scrape/blob/main/subnet-repos/chi-22/datura/tools/tool_manager.py) in the Datura repository.
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
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]

    res = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    return res, ScraperTextRole.SUBNETS_SOURCE_CODE_SUMMARY
