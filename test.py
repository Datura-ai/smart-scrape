import bittensor as bt
from datura.protocol import ScraperStreamingSynapse
import json
import asyncio
import sys  # Import the sys module to access command-line arguments

wallet = bt.wallet(name="validator", hotkey="default")
if len(sys.argv) > 1:
    uids = [int(sys.argv[1])]  # Set uids from the command-line argument
else:
    print("Error: Please provide a user ID as a command-line argument.")
    sys.exit(1)

message = "What are news about in Georgia country?"

axon = bt.axon(wallet=wallet)
dendrite = bt.dendrite(wallet=wallet)
bt.debug(True)

# construct the synapse from the protocol using the message and whatever else we need
synapse = ScraperStreamingSynapse(
    messages=message, model="gpt-4-1106-preview", seed=1234
)

meta = bt.metagraph(netuid=22)
axons = [meta.axons[uid] for uid in uids]


async def main():
    bt.logging.debug(f"Connecting miner uids {uids}")
    async with bt.dendrite(wallet=wallet) as dendrite:
        async_responses = await dendrite.forward(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=120,
            streaming=True,
        )
        full_response = ""
        response_received = False
        for resp in async_responses:
            async for chunk in resp:
                if isinstance(chunk, str):
                    bt.logging.trace(chunk)
                    full_response += chunk
                    response_received = True
            if not response_received:
                bt.logging.debug("No response from miner")
                break
            bt.logging.debug(f"Full_response: {full_response}")
    bt.logging.debug("Finished connecting miner")


if __name__ == "__main__":
    asyncio.run(main())
