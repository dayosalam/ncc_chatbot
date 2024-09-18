import requests

response = requests.post(
    "http://127.0.0.1:8000/ncc_rag/invoke",
    json={'input': {'question': "What factors should be considered when assessing risks?"}}
)

print(response.json()['output'])