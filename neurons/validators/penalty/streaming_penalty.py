import torch
from typing import List
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
from datura.protocol import ScraperStreamingSynapse
import tiktoken
import math


MAX_TOKENS_PER_CHUNK = 2
PENALTY_PER_EXCEEDING_TOKEN = 0.01

HARD_TIMEOUT = 10.0  # Hard timeout in seconds
OVERFLOW_PERIOD = 5.0  # Overflow period in seconds
MAX_PENALTY = 100.0  # Maximum penalty percentage

def calculate_time_penalty(elapsed_time: float, limit: float = HARD_TIMEOUT, overflow: float = OVERFLOW_PERIOD) -> float:

    if elapsed_time <= limit:
        return 0.0
    elif limit < elapsed_time <= limit + overflow:
        # Normalized overflow time 
        normalized = (elapsed_time - limit) / overflow
        # Exponential penalty
        penalty = math.pow(normalized, 2) * MAX_PENALTY
        return min(penalty, MAX_PENALTY)
    else:
        return MAX_PENALTY

class StreamingPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.streaming_penalty.value

    def calculate_penalties(
        self, responses: List[ScraperStreamingSynapse], tasks: List[Task]
    ) -> torch.FloatTensor:
        accumulated_penalties = torch.zeros(len(responses), dtype=torch.float32)
        encoding = tiktoken.get_encoding("cl100k_base")

        for index, response in enumerate(responses):
            streamed_text_chunks = []

            for chunks in response.text_chunks.values():
                streamed_text_chunks.extend(chunks)

            if not streamed_text_chunks:
                # If no chunks received, apply maximum penalty
                accumulated_penalties[index] = 1.0
                continue

            # Token-Based Penalty Calculation
            token_penalty = 0.0
            token_counts = [
                (len(encoding.encode(chunk)) if chunk is not None else 0)
                for chunk in streamed_text_chunks
            ]

            for token_count in token_counts:
                if token_count > MAX_TOKENS_PER_CHUNK:
                    penalty = (token_count - MAX_TOKENS_PER_CHUNK) * PENALTY_PER_EXCEEDING_TOKEN
                    token_penalty = min(1.0, token_penalty + penalty)

            #Time-Based Penalty Calculation
            if response.elapsed_time is not None:
                time_penalty = calculate_time_penalty(
                    elapsed_time=response.elapsed_time,
                    limit=HARD_TIMEOUT,
                    overflow=OVERFLOW_PERIOD
                )
                # Normalize time_penalty to [0,1]
                time_penalty_normalized = time_penalty / MAX_PENALTY
            else:
                # If elapsed_time is not provided, assume maximum penalty
                time_penalty_normalized = 1.0

            #Combine Penalties
            total_penalty = min(1.0, token_penalty + time_penalty_normalized)

            #Assign the Accumulated Penalty
            accumulated_penalties[index] = total_penalty

        return accumulated_penalties


