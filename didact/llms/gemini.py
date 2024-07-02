import asyncio

import google.generativeai as genai
import numpy as np

from didact.llms.base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def embed(self, value: str, num_tries: int = 3) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed(value, num_tries))

    async def aembed(self, value: str, num_tries: int = 3) -> np.ndarray:
        num_tried = 0
        while num_tried < num_tries:
            try:
                response = await genai.embed_content_async(
                    # model="models/embedding-001",
                    model="models/text-embedding-004",
                    content=value,
                    task_type="retrieval_document",
                )
                return np.array(response["embedding"])
            except ValueError:
                pass
            num_tried += 1
        raise ValueError("Error fetching embeddings")

    def prompt(self, value: str, num_tries: int = 3) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aprompt(value, num_tries))

    async def aprompt(self, value: str, num_tries: int = 3) -> str:
        num_tried = 0
        while num_tried < num_tries:
            try:
                model = genai.GenerativeModel(
                    # model_name="gemini-pro",
                    model_name="gemini-1.5-flash",
                )
                response = await model.generate_content_async(value)
                return response.text
            except ValueError:
                pass
            num_tried += 1
        raise ValueError("Error fetching prompt")
