from pathlib import Path

import pypdf


def parse_pdf(path: Path) -> str:
    with open(path, "rb") as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        pages: dict[str, str] = {}
        total_length = 0

        for i, page in enumerate(pdf_reader.pages):
            pages[str(i + 1)] = page.extract_text()
            total_length += len(pages[str(i + 1)])
        
    # TODO - preserve page numbers
    return "\n".join(pages.values())
