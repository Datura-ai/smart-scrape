# import json
# import asyncio
# import random
# from datura.dataset.dataset import MockDiscordQuestionsDataset
# from datura.services.discord_api_wrapper import DiscordAPIClient
# from datura.utils import call_openai
# from datura.protocol import DiscordPromptAnalysisResult
# import bittensor as bt


# discord_api_query_example = {
#     "query": "from:cosmicquantum in:datura,sybil discord api",
#     "limit": "10",
#     "page": "1",
# }

# body_examples = [
#     "from:theiskaa",
#     "from:theiskaa,cosmicquantum",
#     "from:cosmicquantum in:datura,sybil",
#     "from:cosmicquantum, in:datura,sybil after:2023-03-15",
#     "before:2023-03-15",
#     "after:2023-03-15",
#     "during:2023-03-15",
#     "before_days:1d",
#     "phrase here",
#     "in:datura release",
#     "discord api in:general",
#     "in:channel-name",
#     "from:theiskaa before:2023-03-15 some keyword here",
#     "urgent meeting",
#     "from:user1,user2,user3",
#     "in:channel1,channel2,channel3",
#     "in:channel-name search phrase",
#     "after:2023/03/15 in:datura announcements",
#     "from:user1 in:channel1,channel2 before:25/03/2023",
#     "from:user2 after:01/04/2023 project updates",
#     "in:general,random during:2023-02-01",
#     "bug reports in:datura,general,sybil",
# ]

# bad_query_examples = """
#     ""
#     #Explanation: Empty query string.

#     "from:"
#     #Explanation: Empty field for the 'from:' keyword. A username must be provided after the 'from:' keyword.

#     "in:"
#     #Explanation: Empty field for the 'in:' keyword. A channel name must be provided after the 'in:' keyword.

#     "after:march 25"
#     #Explanation: date should be formatted via dd/mm/yyyy format.

#     "before:march 25"
#     #Explanation: date should be formatted via dd/mm/yyyy format.

#     "before_days:two days"
#     #Explanation: before days field shouldn't contain words, it should have format like 1d and 2d.

#     "from:userid123"
#     #Explanation: User IDs cannot be used with the 'from:' keyword. Use usernames instead.

#     "in:serverid456"
#     #Explanation: Server/Guild IDs cannot be used with the 'in:' keyword. Use channel names instead.

#     "from:channel-name"
#     #Explanation: Channel names cannot be used with the 'from:' keyword. Use usernames instead.

#     "in:username-name"
#     #Explanation: User names cannot be used with the 'in:' keyword. Use channel names instead.

#     "Recent announcements in #alpha"
#     #Explanation: The query should be broken down into separate keywords and filters. For recent announcements in the datura channel,
#     it should be 'in:alpha after:<dd/mm/yyy conversion of date representation> announcements'.
# """


# def get_query_gen_prompt(prompt, is_accuracy=True):
#     accuracy_text = ""
#     if is_accuracy:
#         accuracy_text = """
#         RULES:
#             1. Accurately generate keywords, phrases, dates, and 'from:' user mentions and 'in:' channel mentions that are closely related to the user's prompt. After generating these, construct a Discord API query.
#         """
#     else:
#         accuracy_text = """
#         RULES:
#             1. Generate keywords, phrases, dates, and 'from:' user mentions and 'in:' channel mentions that are similar but not necessarily closely related to the user's prompt. After generating these, construct a Discord API query.
#         """

#     content = f"""
#     Given the specific user's prompt:
#     <UserPrompt>
#     '{prompt}'
#     </UserPrompt>

#     Please perform the following tasks and provide the results in a JSON object format:

#     1. Break down the <UserPrompt> into separate keywords, phrases, and modifiers. Discard any unnecessary words or sentence structures.

#     2. Identify and list the key keywords, focusing on specific subjects (e.g., channel names, user mentions, topics).

#     4. Identify and list relevant modifiers (e.g., dates, filters like 'recent', 'latest', 'today', 'yesterday' and etc).,

#     5. Generate Discord API body params based on the refined keywords, channel mentions (if any), user mentions (if any), and dates for a query related to <UserPrompt>. Incorporate filters to ensure relevance and specificity.

#     {accuracy_text}

#     Discord API:
#     1. Body Params Example: {discord_api_query_example}

#     2. Correct Body Params Examples:
#     <CORRECT_EXAMPLES>
#     {body_examples}
#     </CORRECT_EXAMPLES>

#     3. Bad 'body.query' Examples:
#     <BAD_QUERY_EXAMPLES>
#        {bad_query_examples}
#     </BAD_QUERY_EXAMPLES>

#     4. Body Fields Rules:
#         - Set 'limit' to 10 always.

#     5. 'body.query' Rules:
#           - Use 'in:' followed by the channel name if the prompt contains '#channel_name'. For example, if the prompt is 'recent announcements in #general', use 'in:general'.
#           - Use 'from:' followed by the username if the prompt contains '@username'. For example, if the prompt is 'updates from @user1', use 'from:user1'.
#           - If multiple channels are mentioned, separate them with commas (e.g., 'in:channel1,channel2').
#           - If multiple users are mentioned, separate them with commas (e.g., 'from:user1,user2,user3').
#           - Omit 'in:' if no channel mention (starting with '#') is specified.
#           - Omit 'from:' if no username mention (starting with '@') is specified.

#     6. User prompt to generated query example mappings:
#           - "What are the recent announcements in #alpha": "in:alpha announcements"
#           - "What are the recent announcements in #announcements": "in:announcements"
#           - "Tell me the recent news about bittensor": "bittensor news"
#           - "What @professor is asking in subnet 22": "from:professor in:22"
#           - "What is latest release version of Bittensor?": "bittensor release"
#           - "What are the Hyper parameters of subnet 22?": "hyper parameters in:22"
#           - "What people are talking about TAO wallet?": "TAO wallet"
#           - "Axon configurations in translation subnet": "axon config in:translation"
#           - "What are the recent discussions about the new bittensor server update?": "bittensor server update"
#           - "How do I configure my axon for the image classification subnet?": "axon image classification"
#           - "What are people saying about the new Datura tokenomics proposal?": "datura tokenomics"
#           - "Has there been any news on the upcoming Bittensor hackathon?": "bittensor hackathon"
#           - "What are the system requirements for running a full datura node?": "system requirements chi model"
#           - "How can I stake my TAO tokens and earn rewards?": "stake tao tokens"
#           - "What are the latest performance benchmarks for different subnet configurations?": "performance benchmarks days_before:7d"
#           - "Are there any updates on the integration with other AI platforms?": "bittensor integrations"
#           - "What's the best way to contribute to the Bittensor codebase as a developer?": "contribute bittensor codebase"
#           - "What people discussed today?": "days_before:1d"
#           - "How can we deploy a subnet": "subnet deployment or deploy subnet"
#           - "Test network": "test network"
#           - "Which subnets has implementation of Youtube Search tool?": "subnet youtube search integration"
#           - "Which subnets can interact with Google": "subnet google integration"
#           - "Is there any subnet that generates images?": "subnet image generation"
#           - "When testnet will be fixed?": "testnet issue"
#           - "Whats the best image generation tool on bittensor?": "image generation tool"

#     Output Example:
#     {{
#         "body": {{
#             "query": "in:datura after:2023-03-15 announcement",
#             "limit": 10,
#             "page": 1
#         }}
#     }}
#     """
#     bt.logging.trace("get_query_gen_prompt Start   ============================")
#     bt.logging.trace(content)
#     bt.logging.trace("get_query_gen_prompt End   ==============================")
#     return content


# def get_fix_query_prompt(prompt, old_query, error, is_accuracy=True):
#     task = get_query_gen_prompt(prompt=prompt, is_accuracy=is_accuracy)

#     old_query_text = ""
#     if old_query:
#         old_query_text = f"""Your previous query was:
#         <OLD_QUERY>
#         {old_query}
#         </OLD_QUERY>
#         which did not return any results, Please analyse it and make better query."""
#     content = f"""That was task for you:
#     <TASK>
#     {task}
#     <Task>,

#     That was user's promot:
#     <PROMPT>
#     {prompt}
#     </PROMPT>

#     {old_query_text}

#     That was Discord API's result: "{error}"

#     Please, make a new better Output to get better result from Discord API.
#     Output must be as Output example as described in <TASK>.
#     """
#     return content


# class DiscordPromptAnalyzer:
#     def __init__(
#         self,
#         openai_query_model="gpt-3.5-turbo-0125",
#         openai_fix_query_model="gpt-4-1106-preview",
#     ):
#         self.openai_query_model = openai_query_model
#         self.openai_fix_query_model = openai_fix_query_model
#         self.ds_client = DiscordAPIClient()

#     async def generate_query_params_from_prompt(self, prompt, is_accuracy=True):
#         """
#         This function utilizes OpenAI's API to analyze the user's query and extract relevant information such
#         as keywords, phrases, channel and username mentions.
#         """
#         content = get_query_gen_prompt(prompt, is_accuracy)
#         messages = [{"role": "user", "content": content}]
#         bt.logging.trace(content)
#         res = await call_openai(
#             messages=messages,
#             temperature=0.2,
#             model=self.openai_query_model,
#             seed=None,
#             response_format={"type": "json_object"},
#         )
#         response_dict = json.loads(res)
#         bt.logging.trace("generate_query_params_from_prompt Content: ", response_dict)
#         return self.fix_query_dict(response_dict)

#     def fix_query_dict(self, response_dict):
#         if "api_params" in response_dict and "query" in response_dict["body"]:
#             response_dict["body"]["query"] = response_dict["body"]["query"].replace(
#                 "'", '"'
#             )
#         return response_dict

#     async def fix_discord_query(self, prompt, query, error, is_accuracy=True):
#         """
#         This method refines the user's initial query by leveraging OpenAI's API
#         to parse and enhance the query with more precise phrases, keywords, from:username and
#         in:channel-name mentions,
#         aiming to improve the search results from the Discord API.
#         """
#         try:
#             content = get_fix_query_prompt(
#                 prompt=prompt, old_query=query, error=error, is_accuracy=is_accuracy
#             )
#             messages = [{"role": "user", "content": content}]
#             bt.logging.trace(content)
#             res = await call_openai(
#                 messages=messages,
#                 temperature=0.5,
#                 model=self.openai_fix_query_model,
#                 seed=None,
#                 response_format={"type": "json_object"},
#             )
#             response_dict = json.loads(res)

#             return self.fix_query_dict(response_dict)
#         except Exception as e:
#             bt.logging.info(e)
#             return [], None

#     async def search_messages(self, prompt_analysis: DiscordPromptAnalysisResult):
#         return await self.ds_client.search_messages(prompt_analysis.body)

#     async def analyse_prompt_and_fetch_messages(self, prompt):
#         prompt_analysis = DiscordPromptAnalysisResult()
#         try:
#             _, prompt_analysis = await self.generate_and_analyze_query(prompt)

#             result_json, status_code, response_text = await self.search_messages(
#                 prompt_analysis
#             )

#             if status_code in [429, 502, 503, 504]:
#                 bt.logging.warning(
#                     f"analyse_prompt_and_fetch_messages status_code: {status_code} ===========, {response_text}"
#                 )
#                 await asyncio.sleep(random.randint(15, 30))
#                 result_json, status_code, response_text = await self.search_messages(
#                     prompt_analysis
#                 )

#             if status_code == 400:
#                 bt.logging.info(
#                     f"analyse_prompt_and_fetch_messages: Try to fix bad discords api query ============, {response_text}"
#                 )
#                 result_json, status_code, response_text, prompt_analysis = (
#                     await self.retry_with_fixed_query(
#                         prompt=prompt,
#                         old_query=prompt_analysis,
#                         error=response_text,
#                     )
#                 )

#             if status_code != 200:
#                 bt.logging.error(
#                     f"Discord Query ===================================================, {response_text}"
#                 )
#                 raise Exception(f"analyse_prompt_and_fetch_messages: {response_text}")

#             messages = result_json.get("body")

#             if not messages:
#                 bt.logging.info(
#                     "analyse_prompt_and_fetch_messages: No messages found, attempting next query."
#                 )
#                 result_json, status_code, response_text, prompt_analysis = (
#                     await self.retry_with_fixed_query(
#                         prompt,
#                         old_query=prompt_analysis,
#                         is_accuracy=False,
#                     )
#                 )

#             bt.logging.debug(
#                 "Messages fetched ==================================================="
#             )
#             bt.logging.debug(result_json)
#             bt.logging.debug(
#                 "================================================================"
#             )

#             return result_json, prompt_analysis
#         except Exception as e:
#             bt.logging.error(f"analyse_prompt_and_fetch_tweets, {e}")
#             return {"meta": {"result_count": 0}}, prompt_analysis

#     async def generate_and_analyze_query(self, prompt):
#         query = await self.generate_query_params_from_prompt(prompt)
#         prompt_analysis = DiscordPromptAnalysisResult()
#         prompt_analysis.fill(query)
#         self.set_limit(prompt_analysis.body)
#         bt.logging.info(
#             "Discord Query ==================================================="
#         )
#         bt.logging.info(prompt_analysis)
#         bt.logging.info(
#             "================================================================"
#         )
#         return query, prompt_analysis

#     def set_limit(self, body, limit=10):
#         body["limit"] = limit

#     async def retry_with_fixed_query(
#         self, prompt, old_query, error=None, is_accuracy=True
#     ):
#         retry_attempts = 3
#         for attempt in range(retry_attempts):
#             try:
#                 new_query = await self.fix_discord_query(
#                     prompt=prompt, query=old_query, error=error, is_accuracy=is_accuracy
#                 )
#                 prompt_analysis = DiscordPromptAnalysisResult()
#                 prompt_analysis.fill(new_query)
#                 self.set_limit(prompt_analysis.body)
#                 result_json, status_code, response_text = await self.search_messages(
#                     prompt_analysis
#                 )

#                 if status_code == 400:
#                     # TODO: Convert to Exception(response_text)
#                     raise response_text

#                 return result_json, status_code, response_text, prompt_analysis
#             except Exception as e:
#                 bt.logging.info(
#                     f"retry_with_fixed_query Attempt {attempt + 1} failed with error: {e}"
#                 )
#                 # Update the error variable with the current exception for the next retry attempt
#                 error = e
#                 old_query = new_query
#                 if attempt == retry_attempts - 1:
#                     raise e
#                 else:
#                     bt.logging.info(
#                         f"retry_with_fixed_query Retrying... Attempt {attempt + 2}"
#                     )


# if __name__ == "__main__":
#     client = DiscordPromptAnalyzer()

#     dt = MockDiscordQuestionsDataset()
#     questions = []
#     for topic in dt.topics:
#         questions = []
#         for template in dt.question_templates:
#             question = template.format(topic)
#             questions.append(question)

#         results = asyncio.run(
#             asyncio.gather(
#                 *(
#                     client.analyse_prompt_and_fetch_messages(question)
#                     for question in questions
#                 )
#             )
#         )
#         for (result_json, prompt_analysis), qs in zip(results, questions):
#             messages_amount = result_json.get("meta", {}).get("result_count", 0)
#             if messages_amount <= 0:
#                 print(
#                     "=====================START result_json======================================="
#                 )
#                 print(messages_amount, "     ===  ", question)
#                 print("   ")
#                 print(
#                     "=====================START prompt_analysis ======================================="
#                 )
#                 print(prompt_analysis.api_params)
#                 print(
#                     "=====================END prompt_analysis ======================================="
#                 )
#                 print(
#                     "=====================END result_json======================================="
#                 )
