from pathlib import Path
from time import perf_counter
from typing import Sequence
import asyncio
import json

from supabase import create_client, Client
from tqdm import tqdm
import numpy as np

from didact.config import settings
from didact.llms.base import BaseLLM
from didact.llms.gemini import GeminiLLM
from didact.readers import parse_pdf
from didact.stores.supabase_vector_store import DocumentExistsError
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
        raise ValueError("Error inserting document")

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


def record_failed_document(arxiv_json, filename="arxiv-failed.json"):
    with open(filename, "a") as file:
        file.write(json.dumps(arxiv_json) + "\n")


def get_failed_entries(filename="arxiv-failed.json"):
    if not Path(filename).exists():
        return []
    failed_entries = []
    with open(filename, "r") as file:
        for line in file:
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            failed_entries.append(entry_data)
    return failed_entries


def remove_failed_entry(arxiv_json, filename="arxiv-failed.json"):
    if not Path(filename).exists():
        return
    failed_entries = get_failed_entries(filename)
    failed_entries = [d for d in failed_entries if d["id"] != arxiv_json["id"]]
    with open(filename, "w") as file:
        for entry in failed_entries:
            file.write(json.dumps(entry) + "\n")


async def main():
    """
    Not inserted:
    - 0704.0213
    - 0704.0374
    """
    # data_path = "../data/arxiv-metadata-oai-cs-10k.json"
    data_path = "../data/arxiv-metadata-oai-recent-3000.json"
    # data_path = "../data/arxiv-metadata-oai-resnet.json"
    data = []
    with open(data_path, "r") as file:
        for line in file:
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            data.append(entry_data)

    failed_entries = get_failed_entries()
    failed_entry_ids = [d["id"] for d in failed_entries]

    # data = data[287:]
    # for i, row in tqdm(enumerate(data)):
    #     doi = row.get("doi")
    #     arxiv_id = row.get("id")
    #     doi_check_response = supabase.table("documents").select("*").or_(
    #         f"doi.eq.{doi},arxiv_id.eq.{arxiv_id}"
    #     ).execute()
    #     if not len(doi_check_response.data):
    #         print(i)
    #         print(row)
    #         break
    # quit()
    # await store.delete_document(data[0])
    # quit()
    start = perf_counter()
    pbar = tqdm(total=len(data))
    num_ingested = 0
    i = 0
    batchsize = 20
    while True:
        rows = data[i:i+batchsize]
        tasks = [store.aadd_arxiv_json(row, llm) for row in rows]
        for j, coroutine in enumerate(asyncio.as_completed(tasks)):
            try:
                await coroutine
                num_ingested += 1
                if rows[j]["id"] in failed_entry_ids:
                    remove_failed_entry(rows[j])
            except DocumentExistsError:
                pass
            except Exception as e:
                try:
                    await store.delete_document(rows[j])
                except Exception as d:
                    print('Exception: ', d)
                if rows[j]["id"] not in failed_entry_ids:
                    record_failed_document(rows[j])
                print('Exception: ', e)
            pbar.update(1)
        i += batchsize
        if i >= len(data): break
        # break
        if num_ingested >= 949:
            break
    print(f"Average time per document: {(perf_counter() - start) / num_ingested}s")


if __name__ == "__main__":
    asyncio.run(main())
