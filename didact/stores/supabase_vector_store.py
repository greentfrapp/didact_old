from itertools import zip_longest
from pathlib import Path
from typing import Sequence
import asyncio
import json

from supabase._async.client import create_client as create_async_client, AsyncClient
import numpy as np

from didact.bib import doi_to_apa
from didact.llms.base import BaseLLM
from didact.readers import parse_pdf
from didact.types import Embeddable, EmbeddedText
from didact.utils import adownload_arxiv_pdf
from .vector_store import VectorStore


class SupabaseVectorStore(VectorStore):
    texts: list[Embeddable] = []
    _embeddings: np.ndarray | None = None
    
    def __init__(self, supabase_url: str, supabase_service_key: str):
        self.supabase_url = supabase_url
        self.supabase_service_key = supabase_service_key
    
    async def get_async_client(self) -> AsyncClient:
        return await create_async_client(self.supabase_url, self.supabase_service_key)

    def clear(self) -> None:
        self.texts = []
        self._embeddings = None

    def add_arxiv_json(self, arxiv_json, llm: BaseLLM):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aadd_arxiv_json(arxiv_json, llm))

    async def aadd_arxiv_json(self, arxiv_json, llm: BaseLLM, batchsize: int = 30):
        supabase = await self.get_async_client()
        arxiv_id = arxiv_json.get("id")
        title = arxiv_json.get("title")
        abstract = arxiv_json.get("abstract")
        authors = arxiv_json.get("authors")
        doi = arxiv_json.get("doi")
        url = f"https://arxiv.org/abs/{arxiv_id}"
        versions = arxiv_json.get("versions", [{"version": ""}])
        versions = [v["version"] for v in versions]
        version = versions[-1]

        # Check if doi already exists
        doi_check_response = await supabase.table("documents").select("*").or_(
            f"doi.eq.{doi},arxiv_id.eq.{arxiv_id}"
        ).execute()
        if len(doi_check_response.data):
            raise ValueError(f"{doi or arxiv_id} already exists in database")

        filename = await adownload_arxiv_pdf(arxiv_id, version)
        contents = parse_pdf(filename)
        Path(filename).unlink()
        chunks = []
        max_chunk_size = 3000
        overlap = 100
        while len(contents) > max_chunk_size:
            chunks.append(contents[:max_chunk_size])
            contents = contents[max_chunk_size-overlap:]
        if len(contents): chunks.append(contents)
        
        # Insert document
        response = await supabase.table("documents").insert({
            "url": url,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "doi": doi,
            "arxiv_id": arxiv_id,
        }).execute()
        if not len(response.data):
            raise ValueError("Document not inserted")
        doc_id = response.data[0].get("id")
        
        # Insert chunks in batches
        chunk_groups = list(zip_longest(*(iter(chunks),) * batchsize))
        for chunk_group in chunk_groups:
            chunk_group = [c for c in chunk_group if c]
            embeddings = await asyncio.gather(*[
                llm.aembed(chunk) for chunk in chunk_group
            ])
            await supabase.table("chunks").insert([{
                "value": chunk,
                "vector": emb.tolist(),
                "document": doc_id
            } for emb, chunk in zip(embeddings, chunks)]).execute()
        print(f"Embedded and inserted {title}")

    def add_texts_and_embeddings(
        self,
        texts: Sequence[Embeddable],
    ) -> None:
        self.supabase.table("chunks").insert([{
            "value": text.text,
            "vector": text.embedding,
        } for text in texts]).execute()

    def similarity_search(self, client: BaseLLM, query: str, k: int) -> tuple[Sequence[Embeddable], list[float]]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.asimilarity_search(client, query, k))

    async def asimilarity_search(
        self, client: BaseLLM, query: str, k: int
    ) -> tuple[Sequence[Embeddable], list[float]]:
        # k = min(k, len(self.texts))
        if k == 0:
            return [], []

        supabase = await self.get_async_client()

        np_query = await client.aembed(query)

        response = await supabase.rpc('match_documents', {
            "query_embedding": np_query.tolist(),
            "match_threshold": 0,
            "match_count": k,
        }).execute()

        results = []
        for r in response.data: 
            doi = r.get("document")
            year = "20" + r.get("arxiv_id").split(".")[0][:2]
            if doi:
                citation = doi_to_apa(doi)
            else:
                citation = r.get("authors") + f" ({year}) " + r.get("title")
            results.append(
                EmbeddedText(
                    embedding=json.loads(r.get("vector", "None")),
                    text=r.get("value"),
                    source=citation,
                )
            )
        scores = [r.get("similarity") for r in response.data]

        return (results, scores)
