from typing import List, Dict, Optional, Any
from openai import OpenAI
import asyncio
import os
import json
import bittensor as bt
from langchain_openai import ChatOpenAI
from template.tools.base import BaseTool
from template.tools.get_tools import (
    get_all_tools,
    find_toolkit_by_tool_name,
    find_toolkit_by_name,
)
from template.tools.twitter.twitter_toolkit import TwitterToolkit
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from template.protocol import ScraperTextRole
from openai import AsyncOpenAI
from template.tools.response_streamer import ResponseStreamer
from template.protocol import TwitterPromptAnalysisResult

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


TEMPLATE = """Answer the following question as best you can.
User Question: {input}

You have access to the following tools:
{tools}

You can use multiple tools to answer the question. Order of tools does not matter.

Here is example of JSON array format to return. Keep in mind that this is example:
[
  {{
    "action": "Recent Tweets",
    "args": {{
      "query": "AI trends"
    }}
  }},
  {{
    "action": "Web Search",
    "args": {{
      "query": "What are AI trends?"
    }}
  }}
]
"""

prompt_template = PromptTemplate.from_template(TEMPLATE)

client = AsyncOpenAI(timeout=60.0)


class ToolManager:
    model = "gpt-3.5-turbo-1106"
    openai_summary_model: str = "gpt-3.5-turbo-1106"
    all_tools: List[BaseTool]
    manual_tool_names: List[str]
    tool_name_to_instance: Dict[str, BaseTool]

    is_intro_text: bool
    miner: any

    twitter_prompt_analysis: Optional[TwitterPromptAnalysisResult]
    twitter_data: Optional[Dict[str, Any]]

    def __init__(self, prompt, manual_tool_names, send, model, is_intro_text, miner):
        self.prompt = prompt
        self.manual_tool_names = manual_tool_names
        self.miner = miner
        self.is_intro_text = is_intro_text

        self.response_streamer = ResponseStreamer(send=send)
        self.send = send
        self.model = model
        self.openai_summary_model = self.miner.config.miner.openai_summary_model

        self.all_tools = get_all_tools()
        self.tool_name_to_instance = {tool.name: tool for tool in self.all_tools}
        self.twitter_prompt_analysis = None
        self.twitter_data = None

    async def run(self):
        actions = await self.detect_tools_to_use()

        intro_text_task = self.intro_text(
            model="gpt-3.5-turbo",
            tool_names=[action["action"] for action in actions],
        )

        tasks = [asyncio.create_task(self.run_tool(action)) for action in actions]

        await intro_text_task

        toolkit_results = {}

        for completed_task in asyncio.as_completed(tasks):
            result, toolkit_name, tool_name = await completed_task

            if result is not None:
                if toolkit_name == TwitterToolkit().name:
                    toolkit_results[toolkit_name] = result
                else:
                    if toolkit_name not in toolkit_results:
                        toolkit_results[toolkit_name] = ""

                    toolkit_results[
                        toolkit_name
                    ] += f"{tool_name} results: {result}\n\n"

        for toolkit_name, results in toolkit_results.items():
            response, role = await find_toolkit_by_name(toolkit_name).summarize(
                prompt=self.prompt, model=self.model, data=results
            )

            await self.response_streamer.stream_response(response=response, role=role)

        await self.finalize_summary_and_stream(
            self.response_streamer.get_full_text(),
        )

        if self.response_streamer.more_body:
            await self.send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )

    async def detect_tools_to_use(self):
        # If user provided tools manually, use them
        if self.manual_tool_names:
            return [
                {"action": tool_name, "args": self.prompt}
                for tool_name in self.manual_tool_names
            ]

        # Otherwise identify tools to use based on prompt
        # TODO model
        llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.2)
        chain = prompt_template | llm

        tools_description = render_text_description(self.all_tools)

        message = chain.invoke(
            {
                "input": self.prompt,
                "tools": tools_description,
            }
        )

        actions = []

        try:
            actions = json.loads(message.content)
        except json.JSONDecodeError as e:
            print(e)

        return actions

    async def run_tool(self, action: Dict[str, str]):
        tool_name = action.get("action")
        tool_args = action.get("args")
        tool_instance = self.tool_name_to_instance.get(tool_name)

        if not tool_instance:
            return

        bt.logging.info(f"Running tool: {tool_name} with args: {tool_args}")

        tool_instance.tool_manager = self

        result = await tool_instance.ainvoke(tool_args)

        if tool_instance.send_event:
            bt.logging.info(f"Sending event with data from {tool_name} tool")

            await tool_instance.send_event(
                send=self.send,
                response_streamer=self.response_streamer,
                data=result,
            )

        return result, find_toolkit_by_tool_name(tool_name).name, tool_name

    async def intro_text(self, model, tool_names):
        bt.logging.trace("miner.intro_text => ", self.miner.config.miner.intro_text)
        bt.logging.trace("Synapse.is_intro_text => ", self.is_intro_text)
        if not self.miner.config.miner.intro_text:
            return

        if not self.is_intro_text:
            return

        bt.logging.trace("Run intro text")

        tool_names = ", ".join(tool_names)

        content = f"""
        Generate introduction for that prompt: "{self.prompt}".
        You are going to use {tool_names} to fetch information.

        Something like it: "To effectively address your query, my approach involves a comprehensive analysis and integration of relevant Twitter and Google web search data. Here's how it works:

        Question or Topic Analysis: I start by thoroughly examining your question or topic to understand the core of your inquiry or the specific area you're interested in.

        Twitter Data Search: Next, I delve into Twitter, seeking out information, discussions, and insights that directly relate to your prompt.
        Google search: Next, I search Google, seeking out information, discussions, and insights that directly relate to your prompt.

        Synthesis and Response: After gathering and analyzing this data, I compile my findings and craft a detailed response, which will be presented below"        
        
        Output: Just return only introduction text without your comment
        """
        messages = [{"role": "user", "content": content}]
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            stream=True,
        )

        response_streamer = ResponseStreamer(send=self.send)
        await response_streamer.stream_response(
            response=response, role=ScraperTextRole.INTRO, wait_time=0.1
        )

        return response_streamer.get_full_text()

    async def finalize_summary_and_stream(self, information):
        content = f"""
            In <UserPrompt> provided User's prompt (Question).
            In <Information>, provided highlighted key information and relevant links from Twitter and Google Search.
            
            <UserPrompt>
            {self.prompt}
            </UserPrompt>

                Output Guidelines (Tasks):
                1. Summary: Conduct a thorough analysis of <TwitterData> in relation to <UserPrompt> and generate a comprehensive summary.
                2. Relevant Links: Provide a selection of Twitter links that directly correspond to the <UserPrompt>. For each link, include a concise explanation that connects its relevance to the user's question.
                Synthesize insights from both the <UserPrompt> and the <TwitterData> to formulate a well-rounded response. But don't provide any twitter link, which is not related to <UserPrompt>.
                3. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
                4. You would explain how you did retrieve data based on <PromptAnalysis>.

            <Information>
            {information}
            </Information>
        """

        system_message = """As a summary analyst, your task is to provide users with a clear and concise summary derived from the given information and the user's query.

        Output Guidelines (Tasks):
        1. Summary: Conduct a thorough analysis of <Information> in relation to <UserPrompt> and generate a comprehensive summary.

        Operational Rules:
        1. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
        2. Seamless Integration: Avoid explicitly stating "Based on the provided <Information>" in responses. Assume user awareness of the data integration process.
        3. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
        4. Start text with bold text "Summary:".
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content},
        ]

        response = await client.chat.completions.create(
            model=self.openai_summary_model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )

        await self.response_streamer.stream_response(
            response=response, role=ScraperTextRole.FINAL_SUMMARY
        )

        bt.logging.info(
            "================================== Completion Response ==================================="
        )
        bt.logging.info(
            f"{self.response_streamer.get_full_text()}"
        )  # Print the full text at the end
        bt.logging.info(
            "================================== Completion Response ==================================="
        )
