from typing import Sequence

import numpy as np

from didact.llms.base import BaseLLM
from didact.types import Embeddable
from .vector_store import VectorStore


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class NumpyVectorStore(VectorStore):
    texts: list[Embeddable] = []
    _embeddings: np.ndarray | None = None

    def clear(self) -> None:
        self.texts = []
        self._embeddings = None

    def add_texts_and_embeddings(
        self,
        texts: Sequence[Embeddable],
    ) -> None:
        self.texts.extend(texts)
        self._embeddings = np.array([t.embedding for t in self.texts])

    def similarity_search(
        self, client: BaseLLM, query: str, k: int
    ) -> tuple[Sequence[Embeddable], list[float]]:
        k = min(k, len(self.texts))
        if k == 0:
            return [], []

        np_query = np.array(client.embed(query))

        similarity_scores = cosine_similarity(
            np_query.reshape(1, -1), self._embeddings
        )[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        # minus so descending
        # we could use arg-partition here
        # but a lot of algorithms expect a sorted list
        sorted_indices = np.argsort(-similarity_scores)
        return (
            [self.texts[i] for i in sorted_indices[:k]],
            [similarity_scores[i] for i in sorted_indices[:k]],
        )
