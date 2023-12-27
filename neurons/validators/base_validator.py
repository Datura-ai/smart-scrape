from abc import ABC, abstractmethod
import asyncio
import torch

class AbstractNeuron(ABC):
    @abstractmethod
    def check_config(cls, config):
        """Check the configuration."""
        pass

    @abstractmethod
    def add_args(cls, parser):
        """Add arguments to the parser."""
        pass

    @abstractmethod
    def config(cls):
        """Return configuration."""
        pass

    @property
    @abstractmethod
    def subtensor(self):
        """Return subtensor instance."""
        pass

    @property
    @abstractmethod
    def wallet(self):
        """Return wallet instance."""
        pass

    @property
    @abstractmethod
    def metagraph(self):
        """Return metagraph instance."""
        pass

    @property
    @abstractmethod
    def dentrite(self):
        """Return dentrite instance."""
        pass

    @abstractmethod
    async def get_available_uids(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        pass

    @abstractmethod
    async def update_scores(self, scores, wandb_data):
        """Update scores."""
        pass