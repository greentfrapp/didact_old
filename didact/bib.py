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
