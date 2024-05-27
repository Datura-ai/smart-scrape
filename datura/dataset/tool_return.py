from enum import Enum


class ResponseOrder(Enum):
    LINKS_FIRST = "LINK_FIRST"
    SUMMARY_FIRST = "SUMMARY_FIRST"


class ResponseType(Enum):
    SUMMARY_AND_LINKS = "SUMMARY_AND_LINKS"
    ONLY_LINKS = "ONLY_LINKS"
    ONLY_SUMMARY = "ONLY_SUMMARY"