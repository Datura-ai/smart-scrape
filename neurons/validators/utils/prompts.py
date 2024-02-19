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

import re
import random


class BasePrompt:
    r"""Base class for prompts expecting an extractable response."""

    def __init__(self):
        self.template = ""
        self.extract_pattern = ""

    def text(self, *args) -> str:
        r"""Sanitize input strings and format prompt template."""
        sanitized = args
        tags = find_unique_tags(self.template)
        for tag in tags:
            sanitized = [arg.replace(tag, "") for arg in sanitized]

        return self.template.format(*sanitized)

    def extract(self, response: str):
        r"""Search for the extract pattern in the text using regex."""
        result_pattern = re.compile(self.extract_pattern, re.DOTALL)
        result = re.findall(result_pattern, response)

        # If result found, return it.
        if result:
            return result[0]

        # If no result found, return None.
        return None

    def matches_template(self, input_text) -> bool:
        r"""Checks if the input_text matches the first unformatted part of the prompt template."""
        index = self.template.find("{")
        return input_text[:index] == self.template[:index]

class ScoringPrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.extract_pattern = r"\b([0-9]|10)\b"

    def extract_score(self, response: str) -> float:
        r"""Extract numeric score (range 0-10) from prompt response."""
        extraction = self.extract(response)
        if extraction is not None:
            try:
                score = float(extraction)
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0
        return 0

    @staticmethod
    def mock_response():
        r"""Mock responses to a followup prompt, for use in MockDendritePool."""
        return random.choices(
            ["", f"{ random.randint(0, 10) }</Score>"], weights=[1, 9]
        )[0]


class SummaryRelevancePrompt(ScoringPrompt):
    r"""Scores a summary on a scale from 0 to 10, given a context."""

    def __init__(self):
        super().__init__()
        self.template = summary_relevance_scoring_template

class LinkContentPrompt(ScoringPrompt):
    r"""Scores a summary on a scale from 0 to 10, given a context."""

    def __init__(self):
        super().__init__()
        self.template = link_content_relevance_template

def find_unique_tags(input_text: str):
    r"""Find all substrings that match the pattern '<...>'."""
    matches = re.findall("<([^>]*)>", input_text)
    # Return a list of unique matches.
    return list(set(matches))


def extract_score_and_explanation(generated_text):
    # Regular expression to find the last occurrence of "----\n<Score>"
    # and capture everything after it.
    explanation_match = re.search(r'----\n<Score>\n(.*)', generated_text, re.DOTALL | re.MULTILINE)

    if explanation_match:
        # Extract everything after the last "----\n<Score>".
        result = explanation_match.group(1).strip()
    else:
        result = "Explanation not found"

    return result


link_content_relevance_template = """
Evaluate the relevance of the tweet content in response to a specific question. The score is determined based on the level of relevance the tweet content has to the question, with a focus on whether the content mentions keywords or topics related to the question.

Scores can be:
- 0: The tweet content does not mention any of the keywords or topics related to the question, indicating no relevance.
- 5: The tweet content mentions at least one keyword or topic related to the question but does not fully engage with the question's core topics or only does so tangentially.
- 10: The tweet content is highly relevant, mentioning multiple keywords or topics related to the question and engaging with the core topics in a meaningful way.

Instructions for Scoring:
1. Identify keywords or topics from the question that are essential for the answer.
2. Evaluate the tweet content to determine its level of engagement with these keywords or topics.
3. Assign a score based on the criteria above.

<Question>
{}
</Question>

<Tweet Content>
{}
</Tweet Content>

Output:
Generate Score number (0, 5, or 10) based on relevance:
----
<Score>
"""

summary_relevance_scoring_template = """
Evaluate the correctness, relevance, and depth of an answer given a context and question, focusing on the inclusion of Twitter links as supporting evidence. 
Scores range from 0 to 10:
- 0 for answers completely unrelated or incorrect, especially those not addressing the question's topic.
- 2 for answers relevant to the question but lacking any Twitter links as evidence.
- 3-9 for answers that vary in correctness, relevance, and the inclusion of Twitter links, with higher scores reflecting better quality and more relevant evidence.
- 10 for answers that are not only accurate and relevant but also well-supported by Twitter links, fully addressing the question's demands.

Score Examples:
- Score 0: Answer discusses a completely different topic without any relation to the question.
- Score 2: Answer is on topic but does not provide any Twitter links to support its statements.
- Score 6: Provides a partially correct response with some Twitter links, but lacks comprehensive coverage or depth on the topic.
- Score 8: Offers a thorough answer with relevant Twitter links but misses minor details or broader implications.
- Score 10: Fully satisfies the question with accurate, relevant information and substantial evidence from Twitter links.

Additional Scoring Criteria:
- Accuracy and relevance to the question.
- Depth of insight and coverage of the topic.
- Presence and relevance of Twitter links as supporting evidence.

<Question>
{}
</Question>

<Answer>
{}
</Answer>

Output:
Generate Score number and explain with one sentence why assigned that score:
----
<Score>
"""


link_content_relevance_template = """
Evaluate the relevance of the tweet content in response to a specific question. The score is determined based on the level of relevance the tweet content has to the question, with a focus on whether the content mentions keywords or topics related to the question.

Scores can be:
- Assign a score of 0 if the tweet content fails to mention any keywords or topics related to the question, indicating a lack of relevance.
- Assign a score of 5 if the tweet content mentions at least one keyword or topic from the question but either engages with the question's core topics superficially or only tangentially.
- Assign a score of 10 if the tweet content is highly relevant, incorporating multiple keywords or topics from the question and engaging deeply and meaningfully with the question's core topics.

Instructions for Scoring:
- Identify keywords or topics from the question that are essential for the answer.
- Evaluate the tweet content to determine its level of engagement with these keywords or topics.
- Assign a score based on the criteria above.

<Question>
{}
</Question>

<Tweet Content>
{}
</Tweet Content>

Output:
Generate Score number and explain with one sentence why assigned that score:
----
<Score>
"""


# link_content_relevance_template = """
# Evaluate the relevance of the tweet content in response to a specific question. The score is determined based on whether the tweet content mentions at least one keyword or topic related to the question.

# Scores are binary:
# - 0: The tweet content does not mention any of the keywords or topics related to the question.
# - 1: The tweet content mentions at least one keyword or topic related to the question.

# Instructions for Scoring:
# 1. Identify keywords or topics from the question that are essential for the answer.
# 2. Check if the tweet content includes any of these keywords or topics.
# 3. Assign a score based on the criteria above.

# <Question>
# {}
# </Question>

# <Tweet Content>
# {}
# </Tweet Content>

# Output:
# Generate Score number (0 or 1) based on relevance:
# ----
# <Score>
# """