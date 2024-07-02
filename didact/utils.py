from typing import Optional
import asyncio
import subprocess


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def download_arxiv_pdf(id: str, version: str = "", filename: Optional[str] = None):
    folder = id.split(".")[0]
    dst = filename or f"{id}{version}.pdf"
    download_via_gsutil("arxiv-dataset", f"arxiv/arxiv/pdf/{folder}/{id}{version}.pdf", dst)
    return dst


def download_via_gsutil(bucket_name, source_blob_name, destination_file_name):
    subprocess.run([
        "gsutil",
        "cp",
        f"gs://{bucket_name}/{source_blob_name}",
        destination_file_name,
    ])
