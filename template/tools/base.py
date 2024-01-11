from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional

from langchain.tools import BaseTool as LangchainBaseTool
from pydantic import BaseModel, Field, validator

class ToolEnvKeyType(Enum):
    STRING = "string"
    FILE = "file"
    INT = "int"

    def __str__(self):
        return self.value


class ToolEnvKey(BaseModel):
    label: str = Field()
    key: str = Field()
    key_type: ToolEnvKeyType = Field(default=ToolEnvKeyType.STRING)
    is_required: bool = Field(default=False)
    is_secret: bool = Field(default=False)

    @validator("is_secret", "is_required", pre=True, always=True)
    def check_bool(cls, v):
        """Check if the value is a boolean."""
        if v is None:
            return False
        elif isinstance(v, bool):
            return v
        else:
            raise ValueError("Value should be a boolean")

    @validator("key_type", pre=True, always=True)
    def check_key_type(cls, v):
        """Check if the value is a boolean."""
        if v is None:
            return ToolEnvKeyType.STRING
        elif isinstance(v, ToolEnvKeyType):
            return v
        else:
            raise ValueError("key_type should be string/file/integer")


class BaseTool(LangchainBaseTool):
    tool_id: str
    configs: Dict[str, str] = {}
    slug: Optional[str] = None
    toolkit_slug: Optional[str] = None
    data_source_id: Optional[str] = None

    def get_env_key(self, key: str):
        return self.configs.get(key)


class BaseToolkit(BaseModel):
    toolkit_id: str
    name: str
    description: str
    slug: str
    is_active: bool = Field(default=True)

    def get_tools_with_configs(
        self, db, account, settings, agent_with_configs, callback_handler
    ) -> List[BaseTool]:
        # configs = ConfigModel.get_configs(
        #     db=db, query=ConfigQueryParams(toolkit_id=self.toolkit_id), account=account
        # )
        # config_dict = {config.key: config.value for config in configs}
        tools = self.get_tools()

        for tool in tools:
            # tool.configs = config_dict
            tool.toolkit_slug = self.slug
            # tool.settings = settings
            # tool.account = account
            # tool.agent_with_configs = agent_with_configs

            # if callback_handler:
            #     tool.callbacks = [callback_handler]

        return tools

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        # Add file related tools object here
        pass

    @abstractmethod
    def get_env_keys(self) -> List[ToolEnvKey]:
        # Add file related config keys here
        pass
