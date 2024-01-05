import threading
import os
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request
import uvicorn
import bittensor as bt
import traceback
from validator import neuron
from weights import init_wandb, update_weights
from template import QUERY_MINERS
import time
import asyncio

app = FastAPI()
EXPECTED_ACCESS_KEY = os.environ.get('VALIDATOR_ACCESS_KEY')

neu = neuron()

async def run_syn_qs():
    if neu.config.neuron.run_random_miner_syn_qs_interval > 0:
        neu.run(neu.config.neuron.run_random_miner_syn_qs_interval, QUERY_MINERS.RANDOM)

    if neu.config.neuron.run_all_miner_syn_qs_interval > 0:
        time.sleep(120)
        neu.run(neu.config.neuron.run_all_miner_syn_qs_interval, QUERY_MINERS.ALL)

async def response_stream(data):
    try:
        last_message = data['messages'][-1]
        async for response in neu.twitter_validator.organic(last_message):
            yield f"{response}"

    except Exception as e:
        bt.logging.error(f"error in response_stream {traceback.format_exc()}")


@app.post("/analyse-tweets")
async def process_twitter_validatordator(request: Request, data: dict):
    # Check access key
    access_key = request.headers.get("access-key")
    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")
    
    available_uids = await neu.get_available_uids()

    if not available_uids:
        raise HTTPException(status_code=503, detail="Miners are not available")
  
    return StreamingResponse(response_stream(data))

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8005)

if __name__ == "__main__":
    asyncio.get_event_loop().create_task(run_syn_qs())
    run_fastapi()
