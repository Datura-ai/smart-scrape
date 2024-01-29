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
                 device = 'cpu'
                 ):
        if model_provider == ModelProviders.LOCAL.value:
            self.llm = LocalLLM(model_name=model_name,
                                temperature=temperature,
                                device=device)
        else:
            self.llm = OpenAILLM(model_name=model_name,
                                 temperature=temperature)

    def prompt(self, messages, temperature, stream=False, response_format=None, model=None, seed=None): 
        return self.llm.prompt(messages=messages, 
                               temperature=temperature, 
                               model=model, 
                               seed=seed, 
                               response_format=response_format,
                               stream=stream)

class BaseLLM:
    def __init__(self, model_name, temperature, device='cpu'):
        self.model_name = model_name
        self.temperature = temperature
        self.device = device
        self.system_prompt = ""
        self.do_prompt_injection = "Whether to use a custom 'system' prompt instead of the one sent by bittensor."
        self.do_sample = "Whether to use sampling or not (if not, uses greedy decoding)."
        self.max_new_tokens = 256
        
    @abstractmethod
    def prompt(self):
        pass

    @abstractmethod
    def prompt_steaming(self):
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
            use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        bt.logging.info("Model loaded!")

        if self.device != "cpu":
            self.model = self.model.to(self.device)
    
    def _process_history(self, roles: List[str], messages: List[str]) -> str:
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
        for role, message in zip(roles, messages):
            if role == "system":
                if not self.do_prompt_injection or message != message[0]:
                    processed_history += "" + message.strip() + " "
            if role == "Assistant":
                processed_history += "ASSISTANT:" + message.strip() + "</s>"
            if role == "user":
                processed_history += "USER: " + message.strip() + " "
        return processed_history

    def prompt(self, messages: List[any], stream: True) -> str:
        # history = self._process_history(roles=synapse.roles, messages=synapse.messages)
        # prompt = history + "ASSISTANT:"
        prompt = ''
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        completion = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        # Logging input and generation if debugging is active
        bt.logging.debug("Message: " + str(messages))
        bt.logging.debug("Generation: " + str(completion))
        return completion
