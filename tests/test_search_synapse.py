import requests
import json

tools = [
    "Google Search",
    "Google Image Search",
    "Hacker News Search",
    "Reddit Search",
]

query = "What are the recent sport events?"

try:
    response = requests.get(
        "http://localhost:8005/search",
        params={
            "tools": json.dumps(tools),
            "query": query,
            # "uid": 2,
        },
    )
    response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
    data = response.json()
    print(json.dumps(data, indent=2))
except requests.exceptions.RequestException as error:
    print("Error:", error)
