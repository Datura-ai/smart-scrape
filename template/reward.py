# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


import re
import io
import torch
import openai
import typing
import difflib
import asyncio
import logging
import aiohttp
import requests
import numpy as np
from numpy.linalg import norm
import bittensor as bt
from PIL import Image
from typing import List
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import CLIPProcessor, CLIPModel


# ==== TEXT ====

def calculate_text_similarity(text1, text2):
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate the Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity

async def openai_score(openai_answer: str, response: str, weight: float) -> float:
    loop = asyncio.get_running_loop()
    similarity = await loop.run_in_executor(None, calculate_text_similarity, openai_answer, response)
    words_in_response = len(response.split())
    words_in_openai = len(openai_answer.split())
    # linear similarity requirement based on length of response
    min_similarity = max(1 - 0.001 * (words_in_response - 1), 0.75)
    bt.logging.debug(f"similarity for len {words_in_response} / {words_in_openai}: {similarity}, min_similarity is {min_similarity}")

    return weight if similarity >= min_similarity else 0

