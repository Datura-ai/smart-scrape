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


# ==== IMAGES =====

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Could also verify the date from the url
url_regex = (
    r'https://(?:oaidalleapiprodscus|dalleprodsec)\.blob\.core\.windows\.net/private/org-[\w-]+/'
    r'user-[\w-]+/img-[\w-]+\.(?:png|jpg)\?'
    r'st=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'se=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'(?:sp=\w+&)?'
    r'sv=\d{4}-\d{2}-\d{2}&'
    r'sr=\w+&'
    r'rscd=\w+&'
    r'rsct=\w+/[\w-]+&'
    r'skoid=[\w-]+&'
    r'sktid=[\w-]+&'
    r'skt=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'ske=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'sks=\w+&'
    r'skv=\d{4}-\d{2}-\d{2}&'
    r'sig=[\w/%+=]+'
)

    """Check if the URL points to an image asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                return response.status == 200 and 'image' in response.headers.get('Content-Type', '')
    except Exception as e:
        bt.logging.info(f"Error checking URL: {e}")
        return False
    if len(openai_answer) != len(response):
        bt.logging.warning("The number of embeddings in openai_answer and response do not match.")
        return 0

    # Calculate cosine similarity for each pair of embeddings
    cosine_similarities = []
    for oa_emb, resp_emb in zip(openai_answer, response):
        if norm(oa_emb) == 0 or norm(resp_emb) == 0:
            bt.logging.error("One of the embeddings is a zero vector.")
            return 0
        cosine_similarity = np.dot(oa_emb, resp_emb) / (norm(oa_emb) * norm(resp_emb))
        cosine_similarity = min(1.0, max(cosine_similarity, -1.0))  # Clamp the value to the range [-1, 1]

        cosine_similarities.append(cosine_similarity)

    # Average the cosine similarities
    avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
    bt.logging.info(f"Average similarity: {avg_cosine_similarity}")

    # Check against threshold
    if avg_cosine_similarity > threshold: 
        bt.logging.info("Average embeddings cosine similarity exceeds threshold!")
        return weight
    else:
        bt.logging.info(f"Average embeddings cosine similarity does not exceed threshold: {avg_cosine_similarity}")
        return 0