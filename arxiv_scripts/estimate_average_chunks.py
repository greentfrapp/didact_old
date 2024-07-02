from pathlib import Path

from tqdm import tqdm
import numpy as np

from didact.readers import parse_pdf

source_folder = Path("/home/sweekiat/Downloads/research_papers")

num_chunks = []
for pdf_file in tqdm(list(source_folder.glob("*.pdf"))):
    try:
        contents = parse_pdf(pdf_file)
        chunks = []
        max_chunk_size = 3000
        overlap = 100
        while len(contents) > max_chunk_size:
            chunks.append(contents[:max_chunk_size])
            contents = contents[max_chunk_size-overlap:]
        if len(contents):
            chunks.append(contents)
    except:
        pass
    num_chunks.append(len(chunks))

print(num_chunks)
print("Mean", np.mean(num_chunks))
print("Median", np.median(num_chunks))
