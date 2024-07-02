import google.generativeai as genai
import numpy as np

from didact.llms.base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def embed(self, value: str, num_tries: int = 3) -> np.ndarray:
        num_tried = 0
        while num_tried < num_tries:
            try:
                return np.array(genai.embed_content(
                    # model="models/embedding-001",
                    model="models/text-embedding-004",
                    content=value,
                    task_type="retrieval_document",
                )["embedding"])
            except ValueError:
                pass
            num_tried += 1
        raise ValueError("Error fetching embeddings")

    def prompt(self, value: str, num_tries: int = 3) -> str:
        num_tried = 0
        while num_tried < num_tries:
            try:
                model = genai.GenerativeModel(
                    # model_name="gemini-pro",
                    model_name="gemini-1.5-flash",
                )
                response = model.generate_content(value)
                return response.text
            except ValueError:
                pass
            num_tried += 1
        raise ValueError("Error fetching prompt")
