# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from enum import Enum


class RewardModelType(Enum):
    task_validator = "task_validator_filter"
    accuracy_match = "keyword_match_penalty"
    sentence_match_penalty = "sentence_match_penalty"
    summary_relavance_match = "summary_relavance_match"
    link_content_match = "link_content_match"
    search_summary_relevance_match = "search_summary_relevance_match"


class RewardScoringType(Enum):
    summary_relevance_score_template = "summary_relevance_score_template"
    link_content_relevance_template = "link_content_relevance_template"
    search_relevance_score_template = "search_relevance_score_template"


@dataclass(frozen=True)
class DefaultRewardFrameworkConfig:
    """Reward framework default configuration.
    Note: All the weights should add up to 1.0.
    """

    summary_relevance_weight: float = 0.4
    twitter_content_weight: float = 0.5
    web_search_relavance_weight: float = 0.1
