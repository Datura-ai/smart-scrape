from abc import ABC
from typing import List
from template.tools.base import BaseToolkit, BaseTool, ToolEnvKey, ToolEnvKeyType
from .serp_google_search_tool import SerpGoogleSearchTool


class SerpGoogleSearchToolkit(BaseToolkit, ABC):
    name: str = "Google SERP Search Toolkit"
    description: str = "Toolkit containing tools for performing Google SERP search"
    slug: str = "google-serp-search"
    toolkit_id = "fed46dde-ee8e-420b-a1bb-4a161aa01dca"

    def get_tools(self) -> List[BaseTool]:
        return [SerpGoogleSearchTool()]

    def get_env_keys(self) -> List[ToolEnvKey]:
        return [
            ToolEnvKey(
                label="Serp API Key",
                key="SERP_API_KEY",
                key_type=ToolEnvKeyType.STRING,
                is_required=True,
                is_secret=True,
            )
        ]