
from typing import List
import torch
import random
import requests
import os
import asyncio
import bittensor as bt
from template.utils import call_openai

EXPECTED_ACCESS_KEY = os.environ.get('EXPECTED_ACCESS_KEY', 'hello')
URL_SUBNET_18 = os.environ.get('URL_SUBNET_18')

def call_to_subnet_18_scoring(data):
    try:
        if not URL_SUBNET_18:
            bt.logging.warning("Please set the URL_SUBNET_18 environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md")
            return None
        
        headers = {
            "access-key": EXPECTED_ACCESS_KEY,
            "Content-Type": "application/json"
        }
        response = requests.post(url=f"{URL_SUBNET_18}/text-validator/", 
                                 headers=headers, 
                                 json=data)  # Using json parameter to automatically set the content-type to application/json

        if response.status_code in [401, 403]:
            bt.logging.error(f"Connection issue with Subnet 18: {response.text}")
            return {}
        if response.status_code != 200:
              bt.logging.error(f"ERROR connect to Subnet 18: Status code: {response.status_code}")
              return None
        return response
    except Exception as e:
        bt.logging.warning(f"Error calling Subnet 18 scoring: {e}")
        return None

async def get_score_by_openai(messages):
    try:
        query_tasks = []
        for message_dict in messages:  # Iterate over each dictionary in the list
            ((key, message_list),) = message_dict.items()

            async def query_openai(message):
                try:
                    return await call_openai(
                        messages=message,
                        temperature=0.2,
                        model="gpt-3.5-turbo-16k",
                    )
                except Exception as e:
                    print(f"Error sending message to OpenAI: {e}")
                    return ""  # Return an empty string to indicate failure

            task = query_openai(message_list)
            query_tasks.append(task)

        query_responses = await asyncio.gather(*query_tasks, return_exceptions=True)

        result = {}
        for response, message_dict in zip(query_responses, messages):
            if isinstance(response, Exception):
                print(f"Query failed with exception: {response}")
                response = (
                    ""  # Replace the exception with an empty string in the result
                )
            ((key, message_list),) = message_dict.items()
            result[key] = response
        return result
    except Exception as e:
        print(f"Error processing OpenAI queries: {e}")
        return None