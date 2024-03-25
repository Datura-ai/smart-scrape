from openai import AsyncOpenAI
from template.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


SYSTEM_MESSAGE = """
As search data analyst, your task is to provide users with a clear and concise summary derived from the given search data and the user's query.

Tasks:
1. Relevant Links: Provide a selection of search links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
Synthesize insights from both the <UserPrompt> and the <SearchData> to formulate a well-rounded response.
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.

Output Guidelines (Tasks):
1. Relevant Links: Provide a selection of links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
Synthesize insights from both the <UserPrompt> and the <SearchData> to formulate a well-rounded response.
2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.

Operational Rules:
1. No <SearchData> Scenario: If no SearchData is provided, inform the user that current insights related to their topic are unavailable.
2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
3. Seamless Integration: Avoid explicitly stating "Based on the provided <SearchData>" in responses. Assume user awareness of the data integration process.
4. Please separate your responses into sections for easy reading.
5. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
6. Make headers bold using Markdown.
7. Start text with bold text "Search Summary:".
"""


async def summarize_search_data(prompt: str, model: str, data):
    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <SearchData> I fetch data from Google, Youtube or Wikipedia.

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <SearchData>
    {data}
    </SearchData>
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


def prepare_search_data_for_summary(data):
    result = ""

    for tool_name in data.keys():
        if tool_name == "Web Search":
            res = data[tool_name]

            snippets = []

            for organic_result in res.get("organic_results", []):
                snippet_dict = {}
                if "snippet" in organic_result:
                    snippet_dict["snippet"] = organic_result["snippet"]
                if "snippet_highlighted_words" in organic_result:
                    snippet_dict["snippet_highlighted_words"] = organic_result[
                        "snippet_highlighted_words"
                    ]
                if "rich_snippet" in organic_result:
                    snippet_dict["rich_snippet"] = organic_result["rich_snippet"]
                if "rich_snippet_table" in organic_result:
                    snippet_dict["rich_snippet_table"] = organic_result[
                        "rich_snippet_table"
                    ]
                if "link" in organic_result:
                    snippet_dict["link"] = organic_result["link"]

                snippets.append(snippet_dict)

            data[tool_name] = {"type": "organic", "content": snippets}

        result += f"{tool_name} results: {data[tool_name]}\n\n"

    return result
