import json
import asyncio
from starlette.types import Send
from template.protocol import ScraperTextRole
import bittensor as bt


class ResponseStreamer:
    def __init__(self, send: Send) -> None:
        self.buffer = []  # Reset buffer for finalizing data responses
        self.N = 1
        self.full_text = []  # Initialize a list to store all chunks of text
        self.more_body = True
        self.send = send

    async def send_text_event(self, text: str, role: ScraperTextRole):
        text_data_json = json.dumps(
            {"type": "text", "role": role.value, "content": text}
        )

        await self.send(
            {
                "type": "http.response.body",
                "body": text_data_json.encode("utf-8"),
                "more_body": True,
            }
        )

    async def stream_response(self, response, role: ScraperTextRole, wait_time=None):
        await self.send_text_event(text="\n\n", role=role)

        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            self.buffer.append(token)
            self.full_text.append(token)  # Append the token to the full_text list

            if len(self.buffer) == self.N:
                joined_buffer = "".join(self.buffer)
                await self.send_text_event(text=joined_buffer, role=role)

                if wait_time is not None:
                    await asyncio.sleep(wait_time)

                bt.logging.trace(f"Streamed tokens: {joined_buffer}")
                self.buffer = []  # Clear the buffer for the next set of tokens

    def get_full_text(self):
        return "".join(self.full_text)
