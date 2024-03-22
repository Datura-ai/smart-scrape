import json
import asyncio
import random
from template.dataset.dataset import MockDiscordQuestionsDataset
from template.services.discord_api_wrapper import DiscordAPIClient
from template.utils import call_openai
from template.protocol import DiscordPromptAnalysisResult
import bittensor as bt


discord_api_query_example = {
    "query": "from:cosmicquantum in:datura,sybil discord api",
    "limit": "10",
    "page": "1",
}

body_examples = [
    'from:theiskaa',
    'from:theiskaa,cosmicquantum',
    'from:cosmicquantum in:datura,sybil',
    'before:2023-03-15'
    'after:2023-03-15'
    'during:2023-03-15'
    'before_days:1d'
    'phrase here'
    'datura release'
    'api discord datura'
    'in:channel-name'
    'in:server-name,second-server-name',
    'from:theiskaa before:2023-03-15 some keyword here',
    'urgent meeting',
    'from:user1,user2,user3',
    'in:channel1,channel2,channel3'
]

bad_query_examples = """
    ""
    #Explanation: Empty query string
"""


def get_query_gen_prompt(prompt, is_accuracy=True):
    accuracy_text = ""
    if is_accuracy:
        accuracy_text = """
        RULES:
            1. Accurately generate keywords, phrases, dates based solely on text that is unequivocally relevant to the user's prompt and after generate Discord API query
            2. Use 'from:' operator only with usernames, not channel names or IDs. If multiple users, separate them with commas (e.g., 'from:user1,user2,user3').
            3. If no username is provided, omit the 'from:' operator.
            4. Use 'in:' operator only with channel names, not usernames, guild/server names or IDs. If multiple channels, separate them with commas (e.g., 'in:channel1,channel2,channel3').
            5. If no channel name is provided, omit the 'in:' operator.
            6. If date/time is mentioned in the prompt (e.g., recent, today, yesterday), set 'before' and 'after' filters with appropriate dates.
        """
    else:
        accuracy_text = """
        RULES:
            1. Similar Generate generate keywords, phrases, dates and from: user, in: channel mentions that are closely related to the user's prompt and after generate Discord API query
        """
    content = f"""
        Given the specific User's prompt:
        <UserPrompt>
        '{prompt}'
        </UserPromot>

        , please perform the following tasks and provide the results in a JSON object format:

        1. Identify and list the key keywords related to <UserPrompt>, focusing on specific subjects and relevant modifiers.

        4. Generate Discord API body params based on the refined keywords, channel mentions, from mentions (if any) and dates for a query related to <UserPrompt>. Incorporate filters to ensure relevance and specificity.

        {accuracy_text}

        Discord API:
        1. Body Params: "{discord_api_query_example}"

        2. Body Params correct examples:
        <CORRECT_EXAMPLES>
        {body_examples}
        </CORRECT_EXAMPLES>

        3. body.query bad examples:
        <BAD_QUERY_EXAMPES>
           {bad_query_examples}
        </BAD_QUERY_EXAMPES>

        4. body fields rulesplink:
            - limit set always 10
        5. body.query rules:
            - Enclose phrases consisting of two or more words in double quotes (e.g., "Coca Cola"). Do not use single quotes.
            - If no channel name is provided, omit the 'in:' operator.
            - If no username is provided, omit the 'from:' operator.
            - from: operator only include possible usernames, doesn't need to be exact match of username. i.e from:theiskaa
            - in: operator only include possible discord channel names, doesn't need to be exact match. i.e from:datura
            - To construct effective queries, combine search terms using spaces. if user wants
              search in specific channel or from specific user use in: and from: keywrods.

        Output example:
        {{
            "body": {{
                "query": "constructed query based on phrases, keywords, and dates",
                "limit": 10,
                "page": 1,
            }}
        }}"
    """
    bt.logging.trace("get_query_gen_prompt Start   ============================")
    bt.logging.trace(content)
    bt.logging.trace("get_query_gen_prompt End   ==============================")
    return content


def get_fix_query_prompt(prompt, old_query, error, is_accuracy=True):
    task = get_query_gen_prompt(prompt=prompt, is_accuracy=is_accuracy)

    old_query_text = ""
    if old_query:
        old_query_text = f"""Your previous query was:
        <OLD_QUERY>
        {old_query}
        </OLD_QUERY>
        which did not return any results, Please analyse it and make better query."""
    content = f"""That was task for you:
    <TASK>
    {task}
    <Task>,

    That was user's promot:
    <PROMPT>
    {prompt}
    </PROMPT>

    {old_query_text}

    That was Discord API's result: "{error}"

    Please, make a new better Output to get better result from Discord API.
    Output must be as Output example as described in <TASK>.
    """
    return content


class DiscordPromptAnalyzer:
    def __init__(
        self,
        openai_query_model="gpt-3.5-turbo-0125",
        openai_fix_query_model="gpt-4-1106-preview",
    ):
        self.openai_query_model = openai_query_model
        self.openai_fix_query_model = openai_fix_query_model
        self.ds_client = DiscordAPIClient(
            openai_query_model=openai_query_model,
            openai_fix_query_model=openai_fix_query_model,
        )

    async def generate_query_params_from_prompt(self, prompt, is_accuracy=True):
        """
        This function utilizes OpenAI's API to analyze the user's query and extract relevant information such
        as keywords, phrases, channel and username mentions.
        """
        content = get_query_gen_prompt(prompt, is_accuracy)
        messages = [{"role": "user", "content": content}]
        bt.logging.trace(content)
        res = await call_openai(
            messages=messages,
            temperature=0.2,
            model=self.openai_query_model,
            seed=None,
            response_format={"type": "json_object"},
        )
        response_dict = json.loads(res)
        bt.logging.trace("generate_query_params_from_prompt Content: ", response_dict)
        return self.fix_query_dict(response_dict)

    def fix_query_dict(self, response_dict):
        if "api_params" in response_dict and "query" in response_dict["body"]:
            response_dict["body"]["query"] = (response_dict["body"]["query"].replace("'", '"'))
        return response_dict

    async def fix_discord_query(self, prompt, query, error, is_accuracy=True):
        """
        This method refines the user's initial query by leveraging OpenAI's API
        to parse and enhance the query with more precise phrases, keywords, from:username and
        in:channel-name mentions,
        aiming to improve the search results from the Discord API.
        """
        try:
            content = get_fix_query_prompt(
                prompt=prompt, old_query=query, error=error, is_accuracy=is_accuracy
            )
            messages = [{"role": "user", "content": content}]
            bt.logging.trace(content)
            res = await call_openai(
                messages=messages,
                temperature=0.5,
                model=self.openai_fix_query_model,
                seed=None,
                response_format={"type": "json_object"},
            )
            response_dict = json.loads(res)

            return self.fix_query_dict(response_dict)
        except Exception as e:
            bt.logging.info(e)
            return [], None

    async def search_messages(
        self, prompt_analysis: DiscordPromptAnalysisResult
    ):
        return await self.ds_client.search_messages(prompt_analysis.body)

    async def analyse_prompt_and_fetch_messages(self, prompt):
        prompt_analysis = DiscordPromptAnalysisResult()
        try:
            _, prompt_analysis = await self.generate_and_analyze_query(prompt)

            result_json, status_code, response_text = await self.search_messages(prompt_analysis)

            if status_code in [429, 502, 503, 504]:
                bt.logging.warning(
                    f"analyse_prompt_and_fetch_messages status_code: {status_code} ===========, {response_text}"
                )
                await asyncio.sleep(
                    random.randint(15, 30)
                )
                result_json, status_code, response_text = await self.search_messages(
                    prompt_analysis
                )

            if status_code == 400:
                bt.logging.info(
                    f"analyse_prompt_and_fetch_messages: Try to fix bad discords api query ============, {response_text}"
                )
                result_json, status_code, response_text, prompt_analysis = (
                    await self.retry_with_fixed_query(
                        prompt=prompt,
                        old_query=prompt_analysis,
                        error=response_text,
                    )
                )

            if status_code != 200:
                bt.logging.error(
                    f"Discord Query ===================================================, {response_text}"
                )
                raise Exception(f"analyse_prompt_and_fetch_messages: {response_text}")

            messages_amount = result_json.get("meta", {}).get("result_count", 0)

            if messages_amount == 0:
                bt.logging.info(
                    "analyse_prompt_and_fetch_messages: No messages found, attempting next query."
                )
                result_json, status_code, response_text, prompt_analysis = (
                    await self.retry_with_fixed_query(
                        prompt,
                        old_query=prompt_analysis,
                        is_accuracy=False,
                    )
                )

            bt.logging.debug(
                "Messages fetched ==================================================="
            )
            bt.logging.debug(result_json)
            bt.logging.debug(
                "================================================================"
            )

            bt.logging.info(f"Messages fetched amount ============= {messages_amount}")

            return result_json
        except Exception as e:
            bt.logging.error(f"analyse_prompt_and_fetch_tweets, {e}")
            return {"meta": {"result_count": 0}}, prompt_analysis

    async def generate_and_analyze_query(self, prompt):
        query = await self.generate_query_params_from_prompt(prompt)
        prompt_analysis = DiscordPromptAnalysisResult()
        prompt_analysis.fill(query)
        self.set_limit(prompt_analysis.body)
        bt.logging.info(
            "Discord Query ==================================================="
        )
        bt.logging.info(prompt_analysis)
        bt.logging.info(
            "================================================================"
        )
        return query, prompt_analysis

    def set_limit(self, body, limit=10):
        body["limit"] = limit

    async def retry_with_fixed_query(
        self, prompt, old_query, error=None, is_accuracy=True
    ):
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                new_query = await self.fix_discord_query(
                    prompt=prompt, query=old_query, error=error, is_accuracy=is_accuracy
                )
                prompt_analysis = DiscordPromptAnalysisResult()
                prompt_analysis.fill(new_query)
                self.set_limit(prompt_analysis.body)
                result_json, status_code, response_text = await self.search_messages(
                    prompt_analysis
                )

                if status_code == 400:
                    # TODO: Convert to Exception(response_text)
                    raise response_text

                return result_json, status_code, response_text, prompt_analysis
            except Exception as e:
                bt.logging.info(
                    f"retry_with_fixed_query Attempt {attempt + 1} failed with error: {e}"
                )
                # Update the error variable with the current exception for the next retry attempt
                error = e
                old_query = new_query
                if attempt == retry_attempts - 1:
                    raise e
                else:
                    bt.logging.info(
                        f"retry_with_fixed_query Retrying... Attempt {attempt + 2}"
                    )


if __name__ == "__main__":
   client = DiscordPromptAnalyzer()

   dt = MockDiscordQuestionsDataset()
   questions = []
   for topic in dt.topics:
       questions = []
       for template in dt.question_templates:
           question = template.format(topic)
           questions.append(question)

       results = asyncio.run(
           asyncio.gather(
               *(
                   client.analyse_prompt_and_fetch_messages(question)
                   for question in questions
               )
           )
       )
       for (result_json, prompt_analysis), qs in zip(results, questions):
           messages_amount = result_json.get("meta", {}).get("result_count", 0)
           if messages_amount <= 0:
               print(
                   "=====================START result_json======================================="
               )
               print(messages_amount, "     ===  ", question)
               print("   ")
               print(
                   "=====================START prompt_analysis ======================================="
               )
               print(prompt_analysis.api_params)
               print(
                   "=====================END prompt_analysis ======================================="
               )
               print(
                   "=====================END result_json======================================="
               )
