from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> torch.Tensor:
        """Convert text into a vector representation using SentenceTransformers."""
        return self.model.encode(text, convert_to_tensor=True)
