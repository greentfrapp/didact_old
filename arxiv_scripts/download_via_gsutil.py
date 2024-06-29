import subprocess

def download_blob(bucket_name, source_blob_name, destination_file_name):
    subprocess.run([
        "gsutil",
        "cp",
        f"gs://{bucket_name}/{source_blob_name}",
        destination_file_name,
    ])

if __name__ == "__main__":
    download_blob("arxiv-dataset", "arxiv/arxiv/pdf/0704/0704.0001v2.pdf", "./test2.pdf")
