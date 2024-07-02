from typing import Optional

import subprocess


def download_latest_arxiv_pdf(id: str, version: str = "", filename: Optional[str] = None):
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


if __name__ == "__main__":
    download_via_gsutil("arxiv-dataset", "arxiv/arxiv/pdf/0704/0704.0001v2.pdf", "./test2.pdf")
