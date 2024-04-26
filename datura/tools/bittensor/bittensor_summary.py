from openai import AsyncOpenAI
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As a Bittensor Documentation data analyst, your task is to provide users with a clear and concise answer derived from the given Bittensor Documentation conversation and the user's query.

Output Guidelines (Tasks):
1. Analyze the user's prompt and the provided Bittensor Documentation data and write a well-rounded answer that addresses the user's query.

<OutputExample>
**Bittensor Documentation Summary:**

To install Bittensor On macOS and Linux:
- Use the Bash command:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
  ```
- Or use pip3:
  ```bash
  pip3 install bittensor
  ```
- Or install from source by cloning the repo and running:
  ```bash
  python3 -m pip install -e bittensor/
  ```

On Apple Silicon (M1/M2):
- Create a conda virtual environment
- Activate the bittensor conda environment
- Install shtab
- Install Bittensor with `pip3 install bittensor --no-deps`

On Windows:
- Install WSL 2 (Windows Subsystem for Linux)
- Select Ubuntu Linux distribution
- Follow macOS/Linux installation steps within the WSL environment

After installation, verify it worked by using the `btcli --help` command or checking the version by importing bittensor in Python.
</OutputExample>

Operational Rules:
1. No Bittensor Documentation Data Scenario: If no Bittensor documentation data is provided, inform the user that there are no related documentation.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided Bittensor Documentation data" in responses. Assume user awareness of the data integration process.
5. User-Friendly Language: Do not return text like <UserPrompt>; make responses easy to understand for any user.
6. Use Markdown: Make headers bold using Markdown, code blocks with markdown code blocks.
7. Provide Links with Titles: Include links to relevant resources or information. And, always include the current Bittensor document's original URL link for reference.
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
