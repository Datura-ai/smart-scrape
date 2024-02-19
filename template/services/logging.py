import os
import aiohttp

logging_endpoint_url = os.environ.get("LOGGING_ENDPOINT_URL")


async def save_logs(prompt, response, prompt_analysis, data, miner_uid, score):
    if not logging_endpoint_url:
        return

    async with aiohttp.ClientSession() as session:
        await session.post(
            logging_endpoint_url,
            json={
                "prompt": prompt,
                "data": data,
                "response": response,
                "prompt_analysis": prompt_analysis.dict(),
                "miner_uid": miner_uid,
                "score": score,
            },
        )
