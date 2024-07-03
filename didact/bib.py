"""
https://www.kaggle.com/datasets/Cornell-University/arxiv/discussion/496935

The journal ref field is free-form.

Once a paper is published, there may be a doi in the arXiv data.
(All arXiv papers have an arXiv doi, so not the preprint one.)

eg:
https://export.arxiv.org/oai2?verb=GetRecord&metadataPrefix=arXiv&identifier=oai:arXiv.org:1706.03763
10.1088/1751-8121/aaa305

https://www.doi.org/
see resolver, which redirects to:
https://iopscience.iop.org/article/10.1088/1751-8121/aaa305

Maybe you can find what you're looking for, with a doi api.
"""


import requests


def doi_request(doi: str, headers):
    if doi.startswith("http://"):
        url = doi
    else:
        url = "http://dx.doi.org/" + doi
    
    r = requests.get(url, headers=headers)
    return r


def doi_to_apa(doi: str):
    if not doi: return None
    response = doi_request(doi, {
        "accept": "text/x-bibliography; style=apa"
    })
    return response.text
