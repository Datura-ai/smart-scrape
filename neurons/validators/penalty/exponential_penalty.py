import torch
from typing import List
from neurons.validators.utils.tasks import Task
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
import bittensor as bt
import math
from datura.protocol import Model


MAX_PENALTY = 1.0

class ExponentialTimePenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float, model: Model = Model.NOVA):
        super().__init__(max_penalty)
        self.model = model  # Default to Model.NOVA if no model is provided

    @property
    def name(self) -> str:
        return PenaltyModelType.exponential_penalty.value

    def calculate_penalties(
        self, responses: List[bt.Synapse], tasks: List[Task]
    ) -> torch.FloatTensor:
        penalties = torch.zeros(len(responses), dtype=torch.float32)

        from neurons.validators.api import get_max_execution_time #its lazy import, get_max_execution_time should be in utils file
        # Use get_max_execution_time to determine max_execution_time based on the model
        max_execution_time = get_max_execution_time(self.model)

        if max_execution_time is None:
            raise ValueError(f"get_max_execution_time returned None for model {self.model}")

        for index, response in enumerate(responses):
            process_time = getattr(response.dendrite, "process_time", None)

            if process_time <= max_execution_time:
                # Exponential bonus for tasks within the max execution time
                normalized_time = process_time / max_execution_time
                penalty = 1 - math.exp(-normalized_time)  # Exponential decay
                penalties[index] = penalty  # Higher penalty for smaller process times
            else:
                # No penalty for exceeding the max_execution_time
                penalties[index] = 0.0

        return penalties
