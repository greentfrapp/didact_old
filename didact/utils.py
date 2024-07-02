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
    loop = get_loop()
    loop.run_until_complete(adownload_via_gsutil(id, version, filename))


async def adownload_arxiv_pdf(id: str, version: str = "", filename: Optional[str] = None):
    folder = id.split(".")[0]
    dst = filename or f"{id}{version}.pdf"
    await adownload_via_gsutil("arxiv-dataset", f"arxiv/arxiv/pdf/{folder}/{id}{version}.pdf", dst)
    return dst


def download_via_gsutil(bucket_name, source_blob_name, destination_file_name):
    # subprocess.run([
    #     "gsutil",
    #     "cp",
    #     f"gs://{bucket_name}/{source_blob_name}",
    #     destination_file_name,
    # ])
    loop = get_loop()
    loop.run_until_complete(adownload_via_gsutil(bucket_name, source_blob_name, destination_file_name))


async def adownload_via_gsutil(bucket_name, source_blob_name, destination_file_name):
    process = await asyncio.create_subprocess_exec(
        "gsutil",
        "cp",
        f"gs://{bucket_name}/{source_blob_name}",
        destination_file_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
