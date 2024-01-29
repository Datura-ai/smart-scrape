import time
import torch
import argparse
import bittensor as bt
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod
import asyncio
from . import client

class ModelProviders(Enum):
    OPEN_AI = "openai"
    LOCAL = "local"

    def __str__(self) -> str:
        return self.value
    
class LLMManger:
    def __init__(self, 
                 model_provider = ModelProviders.OPEN_AI,
                 model_name = 'gpt-4-1106-preview',
                 temperature = 0.2,
                 device = 'cpu'):
        if model_provider == ModelProviders.LOCAL.value:
            self.llm = LocalLLM(model_name=model_name,
                                temperature=temperature,
                                device=device)
        else:
            self.llm = OpenAILLM(model_name=model_name,
                                 temperature=temperature)

    async def prompt(self, messages, temperature, stream=False, response_format=None, model=None, seed=None):
        if stream:
            # If streaming is enabled, return the stream directly
            return await self.llm.prompt_stream(messages=messages, temperature=temperature, model=model, seed=seed, response_format=response_format)
        else:
            # If streaming is not enabled, return a single response
            response = await self.llm.prompt_regular(messages=messages, temperature=temperature, model=model, seed=seed, response_format=response_format)
            return response  # This is a single response, not an async generator

    async def _prompt_stream(self, messages, temperature, response_format=None, model=None, seed=None):
        # This method handles the streaming logic
        async for response in self.llm.prompt_stream(messages=messages, temperature=temperature, model=model, seed=seed, response_format=response_format):
            yield response

    async def _prompt_regular(self, messages, temperature, response_format=None, model=None, seed=None):
        # This method handles the non-streaming logic
        return await self.llm.prompt_regular(messages=messages, temperature=temperature, model=model, seed=seed, response_format=response_format)

class BaseLLM:
    def __init__(self, model_name, temperature, device='cpu'):
        self.model_name = model_name
        self.temperature = temperature
        self.device = device
        self.system_prompt = ""
        self.do_prompt_injection = ""
        self.do_sample = ""
        self.max_new_tokens = 4096
        
    @abstractmethod
    async def prompt(self):
        pass


class OpenAILLM(BaseLLM):
    def __init__(self, model_name, temperature):
        super(OpenAILLM, self).__init__(model_name, temperature)
        self.model_name = model_name or "gpt-4-1106-preview"

    async def prompt(self, messages, temperature, response_format=None, model=None, seed=None, stream=False):
        model = model or self.model_name
        temperature = temperature or self.temperature
        bt.logging.trace(f"Calling Openai. Temperature = {temperature}, Model = {model}, Seed = {seed},  Messages = {messages}")
        try:
            if stream:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    response_format=response_format,
                    stream=stream
                )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    response_format=response_format
                )
                response = response.choices[0].message.content
                bt.logging.trace(f"validator response is {response}")
                return response
        except Exception as e:
            bt.logging.error(f"Error when calling OpenAI: {e}")
            # await asyncio.sleep(0.5) 
            raise e

class LocalLLM(BaseLLM):
    def __init__(self, model_name, temperature, device):
        """
        Initializes the llmMiner, loading the tokenizer and model based on the given configuration.

        Args:`
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(LocalLLM, self).__init__(model_name, temperature, device)
        bt.logging.info("Loading " + str(self.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=False,
            token="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"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token="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"
        )
        bt.logging.info("Model loaded!")

        if self.device != "cpu":
            self.model = self.model.to(self.device)
    
    def _process_history(self, messages) -> str:
        """
        Processes message history by concatenating roles and messages.

        Args:
            roles (List[str]): A list of roles, e.g., 'system', 'Assistant', 'user'.
            messages (List[str]): A list of corresponding messages for each role.

        Returns:
            str: Processed message history.
        """
        processed_history = ""
        if self.do_prompt_injection:
            processed_history += self.system_prompt
        for item in messages:
            role = item['role']
            message = item['content']
            if role == "system":
                if not self.do_prompt_injection or message != message[0]:
                    processed_history += "" + message.strip() + " "
            if role == "assistant":
                processed_history += "ASSISTANT:" + message.strip() + "</s>"
            if role == "user":
                processed_history += "USER: " + message.strip() + " "
        return processed_history


    async def prompt_stream(self, messages, temperature, response_format=None, model=None, seed=None):
        # This function is for streaming (async generator)
        history = self._process_history(messages=messages)
        prompt = history + "ASSISTANT:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Define a synchronous wrapper function for the generate method
        def generate_sync():
            return self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + self.max_new_tokens,
                temperature=temperature or self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Run the synchronous generate function in an executor
        output = await asyncio.get_event_loop().run_in_executor(None, generate_sync)

        # Yield each token as it's generated
        for token_id in output[0][input_ids.shape[1]:]:
            yield self.tokenizer.decode(token_id, skip_special_tokens=True)

    async def prompt_regular(self, messages, temperature, response_format=None, model=None, seed=None):
        try:
            # This function is for regular async function logic
            history = self._process_history(messages=messages)
            prompt = history + "ASSISTANT:"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + self.max_new_tokens,
                temperature=temperature or self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            completion = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            bt.logging.debug("Generation: " + str(completion))
            return completion
        except Exception as e:
            bt.logging.error(f"analyse_prompt_and_fetch_tweets, {e}")
            raise e
