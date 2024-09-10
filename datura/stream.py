from datura.protocol import (
    ScraperStreamingSynapse,
    extract_json_chunk,
)
import bittensor as bt
import aiohttp
import json
import asyncio
import time
import random


async def collect_response(response: ScraperStreamingSynapse, uid, start_time):
    async for chunk in response:
        if isinstance(chunk, bt.Synapse):
            end_time = time.time()
            duration = end_time - start_time
            process_time = chunk.dendrite.process_time
            if process_time is not None:
                print(
                    f"Miner uid {uid} finished with final synapse after {duration:.2f}s from start time. Dendrite process time: {process_time:.2f}s"
                )
            else:
                print(
                    f"Miner uid {uid} finished with final synapse after {duration:.2f}s from start time. Dendrite process time is None"
                )
            return chunk
    return None


async def collect_responses(async_responses, uids, start_time):
    tasks = [
        asyncio.create_task(collect_response(resp, uid, start_time))
        for resp, uid in zip(async_responses, uids)
    ]

    return await asyncio.gather(*tasks)


async def collect_final_synapses(
    async_responses, uids, start_time, max_execution_time, group_size=15
):
    final_synapses = [None] * len(async_responses)

    if max_execution_time <= 60:
        # Process all async_responses in sequence of groups

        # Split the async_responses into groups of size group_size
        async_responses_groups = [
            async_responses[i : i + group_size]
            for i in range(0, len(async_responses), group_size)
        ]

        group_indices = list(range(len(async_responses_groups)))
        random.shuffle(group_indices)

        for group_index in group_indices:
            async_responses_group = async_responses_groups[group_index]
            group_uids = uids[group_index * group_size : (group_index + 1) * group_size]

            group_final_synapses = await collect_responses(
                async_responses_group, group_uids, start_time
            )

            for i, synapse in enumerate(group_final_synapses):
                final_synapses[group_index * group_size + i] = synapse
    else:
        # Process all async_responses in parallel
        final_synapses = await collect_responses(async_responses, uids, start_time)

    return final_synapses


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
