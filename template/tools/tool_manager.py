from typing import List
from openai import OpenAI
import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from template.tools.serp.serp_google_search_toolkit import SerpGoogleSearchToolkit
from template.tools.twitter.twitter_toolkit import TwitterToolkit
from template.tools.base import BaseTool
from template.tools.get_tools import get_all_tools

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

prompt = hub.pull("hwchase17/react-chat")
os.environ["TAVILY_API_KEY"] = "test"


class ToolManager:
    client = OpenAI()
    model = "gpt-3.5-turbo-1106"
    agent_executor: AgentExecutor

    def __init__(self):
        tools = TwitterToolkit().get_tools()

        agent = create_react_agent(
            llm=ChatOpenAI(model_name="gpt-4"), tools=tools, prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )

    # def set_tools(self, tools):
    #     self.tools = tools

    async def run(self, prompt: str):
        print("Running prompt")
        res = await self.agent_executor.ainvoke({"input": prompt, "chat_history": ""})
        print(res)

    # def run(self, content, tool_choice="auto"):
    #     try:
    #         # Step 1: send the conversation and available functions to the model
    #         messages = [{"role": "user", "content": content}]
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=messages,
    #             tools=self.tools,
    #             tool_choice=tool_choice,  # auto is default, but we'll be explicit
    #         )
    #         response_message = response.choices[0].message
    #         tool_calls = response_message.tool_calls
    #         # Step 2: check if the model wanted to call a function
    #         if tool_calls:
    #             # Step 3: call the function
    #             # Note: the JSON response may not always be valid; be sure to handle errors
    #             messages.append(
    #                 response_message
    #             )  # extend conversation with assistant's reply
    #             # Step 4: send the info for each function call and function response to the model
    #             for tool_call in tool_calls:
    #                 function_name = tool_call.function.name
    #                 function_to_call = self.available_functions[function_name]
    #                 function_args = json.loads(tool_call.function.arguments)
    #                 function_response = function_to_call(
    #                     location=function_args.get("location"),
    #                     unit=function_args.get("unit"),
    #                 )
    #                 messages.append(
    #                     {
    #                         "tool_call_id": tool_call.id,
    #                         "role": "tool",
    #                         "name": function_name,
    #                         "content": function_response,
    #                     }
    #                 )  # extend conversation with function response
    #             second_response = self.client.chat.completions.create(
    #                 model="gpt-3.5-turbo-1106",
    #                 messages=messages,
    #             )  # get a new response from the model where it can see the function response
    #             return second_response
    #     except Exception as err:
    #         print(err)
    #         raise err


if __name__ == "__main__":

    async def run_tool_manager():
        mg = ToolManager()
        await mg.run("What are the latest AI trends on Twitter?")

    run_tool_manager()
