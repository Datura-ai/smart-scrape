from openai import AsyncOpenAI
from datura.dataset.tool_return import ResponseOrder
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


def system_message(response_order: ResponseOrder):
    output_example = ""
    if response_order == ResponseOrder.LINKS_FIRST:
        output_example = """
            Key Sources:
                - [Title and explanation.](https://bbc.com/aw/456)
                - [Title and explanation.](https://bbc.com/w2/123)
            Search Summary:
             Georgia, as a country, hosts a diverse range of sports events catering to various interests. Popular sports in Georgia include football, basketball, rugby union, wrestling, judo, and weightlifting. The sports industry in Georgia is thriving, with a growing interest in modern sports like rugby union, weightlifting, basketball, judo, and football. The country offers a wide array of sporting activities from traditional sports like polo to modern events like football matches, showcasing a rich sporting culture.
        """
    else:
        output_example = """
            Search Summary:
             Georgia, as a country, hosts a diverse range of sports events catering to various interests. Popular sports in Georgia include football, basketball, rugby union, wrestling, judo, and weightlifting. The sports industry in Georgia is thriving, with a growing interest in modern sports like rugby union, weightlifting, basketball, judo, and football. The country offers a wide array of sporting activities from traditional sports like polo to modern events like football matches, showcasing a rich sporting culture.
            Key Sources:
                - [Title and explanation.](https://bbc.com/aw/456)
                - [Title and explanation.](https://bbc.com/w2/123)
        """

    return f"""
    As search data analyst, your task is to provide users with a clear and concise summary derived from the given search data and the user's query.

    Output Guidelines (Tasks):
    1. Key Links: Provide a selection of links that directly correspond to the <UserPrompt>.
    Synthesize insights from both the <UserPrompt> and the <SearchData> to formulate a well-rounded response.
    2. Summarizes key links

    <OutputExample>
    {output_example}
    </OutputExample>

    Operational Rules:
    1. No <SearchData> Scenario: If no SearchData is provided, inform the user that current insights related to their topic are unavailable.
    2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
    3. Seamless Integration: Avoid explicitly stating "Based on the provided <SearchData>" in responses. Assume user awareness of the data integration process.
    4. Please separate your responses into sections for easy reading.
    5. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
    6. Make headers bold using Markdown.
    8. Always return 10 links if available
    9. Do not number the "key Sources"; instead, provide each on a new line.
    10. always maintain the order as shown in <OutputExample>, first providing "Key Sources", followed by "Search Summary".
    11. For each link, include a explanation that connects its relevance to the user's question. The link's description should be 10-25 words, which emphasizes the main topic from that link. [Title and explanation.](https://bbc.com/w2/123)
    """


async def summarize_search_data(prompt: str, model: str, data, response_order):
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
        {"role": "system", "content": system_message(response_order)},
        {"role": "user", "content": content},
    ]

    res = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    return res, ScraperTextRole.SEARCH_SUMMARY


def prepare_search_data_for_summary_old(data):
    result = ""

    for tool_name in data.keys():
        if tool_name == "Google Search":
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
        elif tool_name == "Google Image Search":
            continue

        result += f"{tool_name} results: {data[tool_name]}\n\n"

    return result


def prepare_search_data_for_summary(data):
    standardized_results = []

    # Google Search
    if "Google Search" in data:
        for result in data["Google Search"].get("organic_results", []):
            standardized_results.append(
                {
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "snippet": result.get("snippet"),
                    # 'source': 'Google Search'
                }
            )

    # Google News Search
    if "Google News Search" in data:
        for result in data["Google News Search"].get("news_results", []):
            standardized_results.append(
                {
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "snippet": result.get(
                        "snippet"
                    ),  # Using title as snippet due to lack of snippet in news results
                    # 'source': result.get('source', {}).get('title', 'Unknown')  # Extracting source title, default to 'Unknown'
                }
            )

    # YouTube Search
    if "Youtube Search" in data:
        for result in data.get("Youtube Search", []):
            # Use url_suffix to construct the full YouTube video URL
            video_url = f"https://www.youtube.com{result.get('url_suffix')}"

            # Use the video title as the snippet if 'long_desc' is None or not provided
            snippet = (
                result.get("long_desc")
                if result.get("long_desc")
                else result.get("channel")
            )

            standardized_results.append(
                {
                    "title": result.get("title"),
                    "link": video_url,
                    "snippet": snippet,
                    # 'source': 'Youtube'
                }
            )

    # Arxiv Search
    if "ArXiv Search" in data:
        for result in data.get("ArXiv Search", []):
            standardized_results.append(
                {
                    "title": result.get("title"),
                    "link": result.get("arxiv_url"),
                    "snippet": result.get(
                        "title"
                    ),  # Using title as snippet, assuming no separate snippet available
                    # 'source': 'Arxiv'
                }
            )

    # Wikipedia Search
    if "Wikipedia Search" in data:
        for result in data.get("Wikipedia Search", []):
            standardized_results.append(
                {
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "snippet": result.get("snippet"),
                    # 'source': 'Wikipedia'
                }
            )

    return standardized_results
