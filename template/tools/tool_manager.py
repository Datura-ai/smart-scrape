from typing import List
from openai import OpenAI
from template.tools.base import BaseTool, BaseToolkit
from get_tools import get_all_tools, get_avalaible_functions
import json
import os

OpenAI.api_key = os.environ.get('OPENAI_API_KEY')

class ToolManger:
    client = OpenAI()
    tools:List[BaseToolkit] = get_all_tools()
    available_functions = get_avalaible_functions()
    model = "gpt-3.5-turbo-1106"

    def __int__(self, tools = None, model = None):
        if tools:
            self.tools = tools
        if model:
            self.model = model



    def set_tools(self, tools):
        self.tools = tools
        self.available_functions = {
            # "get_current_weather": (get_current_weather),
        }

    def run(self, content, tool_choice = "auto"):
        # Step 1: send the conversation and available functions to the model
        messages = [{"role": "user", "content": content}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice=tool_choice,  # auto is default, but we'll be explicit
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            messages.append(response_message)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            second_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )  # get a new response from the model where it can see the function response
            return second_response
