from pathlib import Path
from time import perf_counter
from typing import Sequence
import json

from supabase import create_client, Client
import numpy as np

from didact.config import settings
from didact.llms.base import BaseLLM
from didact.llms.gemini import GeminiLLM
from didact.readers import parse_pdf
from didact.stores import SupabaseVectorStore
from didact.types import Embeddable, EmbeddedText
from utils import download_latest_arxiv_pdf


llm = GeminiLLM(settings.GEMINI_API_KEY)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
store = SupabaseVectorStore(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_KEY,
)


def insert_arxiv_doc(arxiv_json):
    arxiv_id = arxiv_json.get("id")
    title = arxiv_json.get("title")
    abstract = arxiv_json.get("abstract")
    authors = arxiv_json.get("authors")
    doi = arxiv_json.get("doi")
    url = f"https://arxiv.org/abs/{arxiv_id}"
    versions = arxiv_json.get("versions", [])
    versions = [v["version"] for v in versions]
    version = ""
    if len(versions) and versions[-1] != "v1":
        version = versions[-1]

    print(f"Inserting {title}")

    response = supabase.table("documents").insert({
        "url": url,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "doi": doi,
        "arxiv_id": arxiv_id,
    }).execute()

    if not len(response.data):
        raise ValueError

    doc_id = response.data[0].get("id")

    filename = download_latest_arxiv_pdf(arxiv_id, version)
    contents = parse_pdf(filename)
    Path(filename).unlink()
    chunks = []
    max_chunk_size = 3000
    overlap = 100
    while len(contents) > max_chunk_size:
        chunks.append(contents[:max_chunk_size])
        contents = contents[max_chunk_size-overlap:]
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        embedding = llm.embed(chunk)
        supabase.table("chunks").insert({
            "value": chunk,
            "vector": embedding.tolist(),
            "document": doc_id
        }).execute()
        # store.add_texts_and_embeddings([EmbeddedText(embedding=embedding, text=chunk)])


if __name__ == "__main__":
    """
    Not inserted:
    - 0704.0213
    - 0704.0374
    """
    data_path = "../data/arxiv-metadata-oai-cs-10k.json"
    # data_path = "../data/arxiv-metadata-oai-resnet.json"
    data = []
    with open(data_path, "r") as file:
        for line in file:
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            data.append(entry_data)
    data = data[20:]
    start = perf_counter()
    # insert_arxiv_doc(data[0])
    num_ingested = 0
    for row in data:
        try:
            store.add_arxiv_json(row, llm)
            num_ingested += 1
        except Exception as e:
            print(e)
        if num_ingested >= 2:
            break
    print((perf_counter() - start) / num_ingested)
