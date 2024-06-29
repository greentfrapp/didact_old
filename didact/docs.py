from pathlib import Path
from pydantic import BaseModel
from typing import List
import re

from didact.llms.base import BaseLLM
from didact.readers import parse_pdf
from didact.stores import NumpyVectorStore, VectorStore
from didact.types import EmbeddedText


class Docs:
    llm: BaseLLM
    store: VectorStore

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.store = NumpyVectorStore()

    def add(self, path: Path):
        """
        1. Parse PDF
        2. Split text into chunks
        3. Embed chunks
        4. Index chunks by embeddings
        """
        contents = parse_pdf(path)
        chunks = []
        max_chunk_size = 3000
        # TODO - overlap chunks
        # TODO - split contents by token instead of characters
        while len(contents) > max_chunk_size:
            chunks.append(contents[:max_chunk_size])
            contents = contents[max_chunk_size:]
        for chunk in chunks:
            embedding = self.llm.embed(chunk)
            self.store.add_texts_and_embeddings([EmbeddedText(embedding=embedding, text=chunk)])

    def summarize_chunk(self, chunk: str, query: str):
        prompt = (
            "Summarize the text below to help answer a question. Do not directly answer the question, instead summarize to give evidence to help answer the question. At the end of your response, provide a score from 1-10 indicating relevance to question. Do not explain your score. Enclose the summary in <summary> tags and enclose the score in <score> tags. If the text is irrelevant, assign a score of 0 and reply <summary>Not applicable</summary><score>0</score>.\n"
            f"\n{chunk}\n\n"
            f"Question: {query}"
        )
        response = self.llm.prompt(prompt)

        summary_rgx_result = re.search("<summary>(?P<summary>.*?)</summary>", response)
        score_rgx_result = re.search("<score>(?P<score>.*?)</score>", response)

        summary = summary_rgx_result.groupdict().get("summary") if summary_rgx_result else None
        score = score_rgx_result.groupdict().get("score") if score_rgx_result else 0
        try:
            score = int(score)
        except ValueError:
            pass
        
        return {
            "summary": summary,
            "score": score,
        }

    def gather_evidence(self, query: str, k: int = 3):
        """
        1. Embed query
        2. Find k candidate chunks
        3. Create prompt with candidate chunks and question
        """
        candidates, scores = self.store.max_marginal_relevance_search(self.llm, query, k, len(self.store.texts))
        summary_score_list = [self.summarize_chunk(c.text, query) for c in candidates]
        return [s["summary"] for s in summary_score_list if s.get("summary") and s.get("score")]

    def ask_llm(self, query: str):
        prompt = (
            "We are collecting background information for the question/task below. Provide a brief summary of information you know (about 50 words) that could help answer the question - do not answer it directly and ignore formatting instructions. It is ok to not answer, if there is nothing to contribute.\n"
            f"\nQuestion: {query}"
        )
        return self.llm.prompt(prompt)

    def answer_question(self, query: str, candidates: List[str], use_ask_llm: bool = False):
        joined_candidates = "\n".join(candidates)
        prompt = "".join((
            "Write an answer for the question below based on the provided context. If the context provides insufficient information, reply \"I cannot answer\".\n"
            # TODO - implement citations
            # "For each part of your answer, indicate which sources most support it via valid citation markers at the end of sentences, like (Example2012).\n"
            "Answer in an unbiased, comprehensive, and scholarly tone. If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences.\n"
            f"\n{joined_candidates}\n\n",
            (f"\nExtra background information: {self.ask_llm(query)}\n\n" if use_ask_llm else ""),
            f"\nQuestion: {query}",
        ))
        return self.llm.prompt(prompt)
    
    def query(self, query: str, k: int = 3):
        candidates = self.gather_evidence(query, k)
        response = self.answer_question(query, candidates)
        return response
