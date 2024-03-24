import re

from typing import List


class WebSearchUtils:
    @staticmethod
    def find_links(text: str) -> List[str]:
        link_regex = r"\[.*?\]\((https?://[^\s\)]+)\)"
        links = []
        for match in re.finditer(link_regex, text):
            link = match.group(1)
            links.append(link)
        return links
