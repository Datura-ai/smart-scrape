from typing import List
from openai import OpenAI
import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from template.tools.serp.serp_google_search_toolkit import SerpGoogleSearchToolkit
from template.tools.twitter.twitter_toolkit import TwitterToolkit
from template.tools.base import BaseTool
from template.tools.get_tools import get_all_tools
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
import json

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

prompt = hub.pull("hwchase17/react-chat")
os.environ["TAVILY_API_KEY"] = "test"


TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:
{tools}

You can use multiple tools to answer the question. Order of tools does not matter.

Here is example of JSON array format to return. Keep in mind that this is example:
[
  {{
    "action": "Get Recent Tweets",
    "args": {{
      "query": "AI trends"
    }}
  }},
  {{
    "action": "Serp Google Search",
    "args": {{
      "query": "What are AI trends?"
    }}
  }}
]

User Question: {input}
"""

prompt_template = PromptTemplate.from_template(TEMPLATE)


class ToolManager:
    client = OpenAI()
    model = "gpt-3.5-turbo-1106"
    agent_executor: AgentExecutor

    def __init__(self):

        # tools = TwitterToolkit().get_tools()
        tools = get_all_tools()

        agent = create_react_agent(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0.2, streaming=True),
            tools=tools,
            prompt=prompt,
        )

        # self.agent_executor = AgentExecutor(
        #     agent=agent,
        #     tools=tools,
        #     verbose=True,
        #     handle_parsing_errors=True,
        #     return_intermediate_steps=True,
        # )

    # def set_tools(self, tools):
    #     self.tools = tools

    async def run(self, prompt: str):
        # res = await self.agent_executor.ainvoke({"input": prompt, "chat_history": ""})

        llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
        chain = prompt_template | llm
        tools = get_all_tools()

        tool_name_to_instance = {tool.name: tool for tool in tools}

        tools_description = render_text_description(tools)

        message = chain.invoke(
            {
                "input": prompt,
                "tools": tools_description,
            }
        )

        actions = []

        try:
            actions = json.loads(message.content)

            for action in actions:
                tool_name = action.get("action")
                tool_args = action.get("args")
                tool = tool_name_to_instance[tool_name]

                tool_res = await tool.ainvoke(tool_args)

                print(tool_res)
        except json.JSONDecodeError as e:
            print(e)

        return actions

        # chunks = []
        # async for chunk in self.agent_executor.astream(
        #     {"input": prompt, "chat_history": ""}
        # ):
        #     steps = chunk.get("steps")

        #     if steps:
        #         observation = steps[0].observation
        #         print("\n\nTOOL OBSERVATION: ", observation, "\n\n")

        #         # TODO Summarize data based on tool (summarize_twitter_data, summarize_serp_google_search_data)

        #         # TODO Send the summarized data to ui
        #     chunks.append(chunk)
        #     print("---------")

        # return ""

    # Uses OpenAI parallel function calling to call tools
    def run_old(self, content, tool_choice="auto"):
        from operator import itemgetter
        from typing import Union

        from langchain.output_parsers import JsonOutputToolsParser
        from langchain_core.runnables import (
            Runnable,
            RunnableLambda,
            RunnableMap,
            RunnablePassthrough,
        )
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4-0125-preview")
        tools = get_all_tools()
        model_with_tools = model.bind_tools(tools)
        tool_map = {tool.name: tool for tool in tools}

        def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
            """Function for dynamically constructing the end of the chain based on the model-selected tool."""
            tool = tool_map[tool_invocation["type"]]
            return RunnablePassthrough.assign(output=itemgetter("args") | tool)

        # .map() allows us to apply a function to a list of inputs.
        call_tool_list = RunnableLambda(call_tool).map()
        chain = model_with_tools | JsonOutputToolsParser() | call_tool_list

        return chain.ainvoke(content)

        # try:
        #     # Step 1: send the conversation and available functions to the model
        #     messages = [{"role": "user", "content": content}]
        #     response = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=messages,
        #         # tools=sesport newslf.tools,
        #         tool_choice=tool_choice,  # auto is default, but we'll be explicit
        #     )
        #     response_message = response.choices[0].message
        #     tool_calls = response_message.tool_calls
        #     # Step 2: check if the model wanted to call a function
        #     if tool_calls:
        #         # Step 3: call the function
        #         # Note: the JSON response may not always be valid; be sure to handle errors
        #         messages.append(
        #             response_message
        #         )  # extend conversation with assistant's reply
        #         # Step 4: send the info for each function call and function response to the model
        #         for tool_call in tool_calls:
        #             function_name = tool_call.function.name
        #             function_to_call = self.available_functions[function_name]
        #             function_args = json.loads(tool_call.function.arguments)
        #             function_response = function_to_call(
        #                 location=function_args.get("location"),
        #                 unit=function_args.get("unit"),
        #             )
        #             messages.append(
        #                 {
        #                     "tool_call_id": tool_call.id,
        #                     "role": "tool",
        #                     "name": function_name,
        #                     "content": function_response,
        #                 }
        #             )  # extend conversation with function response
        #         second_response = self.client.chat.completions.create(
        #             model="gpt-3.5-turbo-1106",
        #             messages=messages,
        #         )  # get a new response from the model where it can see the function response
        #         return second_response
        # except Exception as err:
        #     print(err)
        #     raise err


if __name__ == "__main__":

    async def run_tool_manager():
        mg = ToolManager()
        await mg.run("What are the latest AI trends on Twitter?")

    run_tool_manager()
