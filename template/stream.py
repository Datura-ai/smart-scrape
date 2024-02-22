from template.protocol import ScraperStreamingSynapse, TwitterPromptAnalysisResult
import bittensor as bt
import aiohttp
import json
import asyncio

def extract_json_chunk(chunk):
    stack = []
    start_index = None
    json_objects = []

    for i, char in enumerate(chunk):
        if char == "{":
            if not stack:
                start_index = i
            stack.append(char)
        elif char == "}":
            stack.pop()
            if not stack and start_index is not None:
                json_str = chunk[start_index : i + 1]
                try:
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                    start_index = None
                except json.JSONDecodeError as e:
                    # Handle the case where json_str is not a valid JSON object
                    continue

    remaining_chunk = chunk[i + 1 :] if start_index is None else chunk[start_index:]

    return json_objects, remaining_chunk

async def process_async_responses(async_responses):
    tasks = [collect_generator_results(resp) for resp in async_responses]
    responses = await asyncio.gather(*tasks)
    for response in responses:
        stream_text = ''.join([chunk[1] for chunk in response if not chunk[0]])
        if stream_text:
            yield stream_text  # Yield stream text as soon as it's available
        # Instead of returning, yield final synapse objects with a distinct flag
        final_synapse = next((chunk[1] for chunk in response if chunk[0]), None)
        if final_synapse:
            yield (True, final_synapse)  # Yield final synapse with a flag

async def collect_generator_results(response):
    results = []
    async for result in process_single_response(response):
        results.append(result)
    return results

async def process_single_response(response):
    synapse = ScraperStreamingSynapse(
        messages='', model='', seed=1
    )
    completion = ""
    prompt_analysis = None
    miner_tweets = []

    try:
        async for chunk in response:
            if isinstance(chunk, bt.Synapse):
                    synapse = chunk
                    if chunk.is_failure:
                        raise Exception("Dendrite's status code indicates failure")
            else: 
                chunk_str = chunk.decode("utf-8")
                try:
                    json_objects, remaining_chunk = extract_json_chunk(chunk_str)
                    for json_data in json_objects:
                        content_type = json_data.get("type")

                        if content_type == "text":
                            text_content = json_data.get("content", "")
                            completion += text_content
                            yield (False, text_content)

                        elif content_type == "prompt_analysis":
                            prompt_analysis_json = json_data.get("content", "{}")
                            prompt_analysis = TwitterPromptAnalysisResult()
                            prompt_analysis.fill(prompt_analysis_json)

                        elif content_type == "tweets":
                            tweets_json = json_data.get("content", "[]")
                            miner_tweets = tweets_json
                except json.JSONDecodeError as e:
                    bt.logging.info(f"process_single_response json.JSONDecodeError: {e}")
           
    except Exception as exception:
        if isinstance(exception, aiohttp.ClientConnectorError):
            synapse.dendrite.status_code = "503"
            synapse.dendrite.status_message = f"Service at {synapse.axon.ip}:{str(synapse.axon.port)}/{synapse.__class__.__name__} unavailable."
        elif isinstance(exception, asyncio.TimeoutError):
            synapse.dendrite.status_code = "408"
            synapse.dendrite.status_message = (
                f"Timedout after {synapse.timeout} seconds."
            )
        else:
            synapse.dendrite.status_code = "422"
            synapse.dendrite.status_message = (
                f"Failed to parse response object with error: {str(exception)}"
            )
        bt.logging.debug(f"Process Single Response Combined: {str(exception)}")
        yield (True, synapse)  # Indicate this is the final value to return
        return 
    except GeneratorExit:
        bt.logging.warning(f"Handle it here: GeneratorExit")
        # Handle generator cleanup here
        return
    finally:
        if completion:
            synapse.completion = completion
        if miner_tweets:
            synapse.miner_tweets = miner_tweets
        if prompt_analysis:
            synapse.prompt_analysis = prompt_analysis

        yield (True, synapse)  # Final value