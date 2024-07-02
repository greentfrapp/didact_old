from abc import ABC, abstractmethod
from typing import Any, Sequence

from pydantic import Field
import numpy as np

from didact.types import Embeddable


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class VectorStore(ABC):
    mmr_lambda: float = 0.9

    @abstractmethod
    def add_texts_and_embeddings(
        self,
        texts: Sequence[Embeddable],
    ) -> None:
        pass

    @abstractmethod
    async def similarity_search(
        self, client: Any, query: str, k: int
    ) -> tuple[Sequence[Embeddable], list[float]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def max_marginal_relevance_search(
        self, client: Any, query: str, k: int, fetch_k: int
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        texts, scores = self.similarity_search(client, query, fetch_k)
        
        if len(texts) <= k or self.mmr_lambda >= 1.0:
            return texts, scores

        embeddings = np.array([t.embedding for t in texts])
        np_scores = np.array(scores)
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        selected_indices = [0]
        remaining_indices = list(range(1, len(texts)))

        while len(selected_indices) < k:
            selected_similarities = similarity_matrix[:, selected_indices]
            max_sim_to_selected = selected_similarities.max(axis=1)

            mmr_scores = (
                self.mmr_lambda * np_scores
                - (1 - self.mmr_lambda) * max_sim_to_selected
            )
            mmr_scores[selected_indices] = -np.inf  # Exclude already selected documents

            max_mmr_index = mmr_scores.argmax()
            selected_indices.append(max_mmr_index)
            remaining_indices.remove(max_mmr_index)

        return [texts[i] for i in selected_indices], [
            scores[i] for i in selected_indices
        ]
