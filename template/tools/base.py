from abc import abstractmethod
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool as LangChainBaseTool
from template.protocol import ScraperTextRole


class BaseTool(LangChainBaseTool):
    tool_id: str
    slug: Optional[str] = None
    tool_manager: Any = None

    @abstractmethod
    async def send_event(self, send, response_streamer, data):
        pass


class BaseToolkit(BaseModel):
    toolkit_id: str
    name: str
    description: str
    slug: str
    is_active: bool = Field(default=True)

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        pass

    @abstractmethod
    async def summarize(self, prompt, model, data) -> Tuple[Any, ScraperTextRole]:
        pass
