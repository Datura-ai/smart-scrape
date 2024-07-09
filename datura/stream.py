from datura.protocol import (
    ScraperStreamingSynapse,
    TwitterPromptAnalysisResult,
    extract_json_chunk,
)
import bittensor as bt
import aiohttp
import json
import asyncio
import time


async def process_async_responses(async_responses, uids, start_time):
    tasks = [collect_generator_results(resp) for resp in async_responses]
    responses = await asyncio.gather(*tasks)
    for uid, response in zip(uids, responses):
        final_synapse = next(
            (chunk for chunk in response if isinstance(chunk, bt.Synapse)), None
        )
        if final_synapse:
            end_time = time.time()
            duration = end_time - start_time
            process_time = final_synapse.dendrite.process_time
            if process_time is not None:
                bt.logging.debug(
                    f"Miner uid {uid} finished with final synapse after {duration:.2f}s from start time. Dendrite process time: {process_time:.2f}s"
                )
            else:
                bt.logging.debug(
                    f"Miner uid {uid} finished with final synapse after {duration:.2f}s from start time. Dendrite process time is None"
                )
            yield final_synapse  # Yield final synapse
        else:
            stream_text = "".join(
                [str(chunk) for chunk in response if chunk is not None]
            )
            if stream_text:
                yield stream_text  # Yield stream text as soon as it's available


async def collect_generator_results(response):
    results = []
    async for result in response:
        results.append(result)
    return results


# async def collect_generator_results(response):
#     results = []
#     async for result in process_single_response(response):
#         results.append(result)
#     return results


async def process_single_response(response):
    synapse = ScraperStreamingSynapse(messages="", model="", seed=1)
    completion = ""
    # prompt_analysis = None
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

                        # elif content_type == "prompt_analysis":
                        #     prompt_analysis_json = json_data.get("content", "{}")
                        #     prompt_analysis = TwitterPromptAnalysisResult()
                        #     prompt_analysis.fill(prompt_analysis_json)

                        elif content_type == "tweets":
                            tweets_json = json_data.get("content", "[]")
                            miner_tweets = tweets_json
                except json.JSONDecodeError as e:
                    bt.logging.info(
                        f"process_single_response json.JSONDecodeError: {e}"
                    )

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
        # if prompt_analysis:
        #     synapse.prompt_analysis = prompt_analysis

        yield (True, synapse)  # Final value
