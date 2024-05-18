import requests

url = "http://localhost:8000/stream/"

with requests.get(url, stream=True) as r:
    for chunk in r.iter_content(None, decode_unicode=True):
        if chunk:
            print(chunk, end='', flush=True)