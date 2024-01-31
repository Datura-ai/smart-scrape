import bittensor as bt
from template.protocol import TwitterScraperStreaming
import json
import asyncio

wallet = bt.wallet(name = "validator", hotkey = "default")
uids = [221]
message = "tell me about sports"

axon = bt.axon(wallet=wallet)
dendrite = bt.dendrite(wallet=wallet)

# construct the synapse from the protocol using the message and whatever else we need
synapse = TwitterScraperStreaming(messages=message, model="gpt-4-1106-preview", seed="1234")

meta = bt.metagraph(netuid = 22)
axons = [meta.axons[uid] for uid in uids]

async def fetch_responses(axons, synapse):
    async_responses = dendrite.query(
        axons=axons, 
        synapse=synapse, 
        deserialize=False,
        timeout=120,
        streaming=True,
    )
    return async_responses

async def process_single_response(resp, prompt):
    default = TwitterScraperStreaming(messages=prompt, model='', seed=123)
    full_response = ""
    synapse_object = None

    try:
        async for chunk in resp:
            if isinstance(chunk, str):
                full_response += chunk
            elif isinstance(chunk, bt.Synapse):
                if chunk.is_failure:
                    raise Exception("Dendrite's status code indicates failure")
                synapse_object = chunk
    except Exception as e:
        bt.logging.trace(f"Process Single Response: {e}")
        return default

    if synapse_object is not None:
        return synapse_object

    return default
    
async def process_async_responses(async_responses, prompt):
    # Create a list of coroutine objects for each response
    tasks = [process_single_response(resp, prompt) for resp in async_responses]
    # Use asyncio.gather to run them concurrently
    responses = await asyncio.gather(*tasks)
    return responses

async def main():
    async_responses = await fetch_responses(axons, synapse)
    responses = await process_async_responses(async_responses, message)
    for response in responses:
        print(response)

if __name__ == "__main__":
    asyncio.run(main())



    