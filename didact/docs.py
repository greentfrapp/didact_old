from pathlib import Path
from pydantic import BaseModel
from time import perf_counter
from typing import List
import asyncio
import re

from didact.llms.base import BaseLLM
from didact.readers import parse_pdf
from didact.stores import NumpyVectorStore, VectorStore
from didact.types import EmbeddedText


class Docs:
    llm: BaseLLM
    store: VectorStore

    def __init__(self, llm: BaseLLM, vector_store: VectorStore = None):
        self.llm = llm
        self.store = vector_store or NumpyVectorStore()

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
        if len(contents):
            chunks.append(contents)
        for chunk in chunks:
            embedding = self.llm.embed(chunk)
            self.store.add_texts_and_embeddings([EmbeddedText(embedding=embedding, text=chunk)])

    def summarize_chunk(self, chunk: str, query: str, source: str = None):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.asummarize_chunk(chunk, query, source))

    async def asummarize_chunk(self, chunk: str, query: str, source: str = None):
        prompt = (
            "Summarize the text below to help answer a question. Do not directly answer the question, instead summarize to give evidence to help answer the question. At the end of your response, provide a score from 1-10 indicating relevance to question. Do not explain your score. Enclose the summary in <summary> tags and enclose the score in <score> tags. If the text is irrelevant, assign a score of 0 and reply <summary>Not applicable</summary><score>0</score>.\n"
            f"\n{chunk}\n\n"
            f"Question: {query}"
        )
        response = await self.llm.aprompt(prompt)

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
            "source": source,
        }

    def gather_evidence(self, query: str, k: int = 5, fetch_k: int = 10, summarize: bool = True, verbose: bool = False):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.agather_evidence(query, k, fetch_k, summarize, verbose))

    async def agather_evidence(self, query: str, k: int = 5, fetch_k: int = 10, summarize: bool = True, verbose: bool = False):
        """
        1. Embed query
        2. Find k candidate chunks
        3. Create prompt with candidate chunks and question
        """
        start = perf_counter()
        candidates, scores = await self.store.amax_marginal_relevance_search(self.llm, query, k, fetch_k)
        if verbose:
            print(f"Gather evidence: {perf_counter() - start}s")
        start = perf_counter()
        if summarize:
            summary_score_list = await asyncio.gather(*[self.asummarize_chunk(c.text, query, c.source) for c in candidates])
        else:
            summary_score_list = [{"summary": c.text, "source": c.source, "score": 5} for c in candidates]
        if verbose and summarize:
            print(f"Summarize chunks: {perf_counter() - start}s")
        return [s["summary"] + "\n(" + s["source"] + ")" for s in summary_score_list if s.get("summary") and s.get("score")]

    def ask_llm(self, query: str, verbose: bool = False):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aask_llm(query, verbose))

    async def aask_llm(self, query: str, verbose: bool = False):
        start = perf_counter()
        prompt = (
            "We are collecting background information for the question/task below. Provide a brief summary of information you know (about 50 words) that could help answer the question - do not answer it directly and ignore formatting instructions. It is ok to not answer, if there is nothing to contribute.\n"
            f"\nQuestion: {query}"
        )
        response = await self.llm.aprompt(prompt)
        if verbose:
            print(f"Ask LLM: {perf_counter() - start}s")
        return response

    def answer_question(self, query: str, candidates: List[str], background_info: str = None):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.answer_question(query, candidates, background_info))

    async def aanswer_question(self, query: str, candidates: List[str], background_info: str = None):
        joined_candidates = "\n".join(candidates)
        prompt = "".join((
            "Write an answer for the question below based on the provided context. If the context provides insufficient information, reply \"I cannot answer\".\n"
            # TODO - improve citations
            "For each part of your answer, indicate which sources most support it via valid citation markers at the end of sentences, like (Foo et al., 2020).\n"
            "Answer in an unbiased, comprehensive, and scholarly tone. If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences.\n"
            f"\n{joined_candidates}\n\n",
            (f"\nExtra background information: {background_info}\n\n" if background_info else ""),
            f"\nQuestion: {query}",
        ))
        return await self.llm.aprompt(prompt)

    def query(self, query: str, k: int = 5, fetch_k: int = 10, summarize: bool = True, use_ask_llm: bool = False, verbose: bool = False):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aquery(query, k, fetch_k, summarize, use_ask_llm, verbose))
    
    async def aquery(self, query: str, k: int = 5, fetch_k: int = 10, summarize: bool = True, use_ask_llm: bool = False, verbose: bool = False):
        if use_ask_llm:
            candidates, background_info = await asyncio.gather(
                self.agather_evidence(query, k, fetch_k, summarize=summarize, verbose=verbose),
                self.aask_llm(query, verbose),
            )
        else:
            candidates = await self.agather_evidence(query, k, fetch_k, summarize=summarize, verbose=verbose)
            background_info = None
        # if verbose:
        #     print(candidates)
        start = perf_counter()
        response = await self.aanswer_question(query, candidates, background_info=background_info)
        if verbose:
            print(f"Answer question: {perf_counter() - start}s")
        return response
