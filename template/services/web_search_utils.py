import re
from typing import List


class WebSearchUtils:
    @staticmethod
    def find_links(text: str) -> List[str]:
        link_regex = r"(?:^|\n)\s*(?:\d+\s*)?([^\n]*?)\s*\[([^\]]*?)\]\((.*?)\)"
        links = []

        for match in re.finditer(link_regex, text, re.MULTILINE):
            link = match.group(3)
            links.append(link)

        return links
