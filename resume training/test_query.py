import requests

# Define the endpoint URL
url = 'http://127.0.0.1:5000/query'

# Define the query you want to test
query = {
    "query": "what is the definition of nbcc"
}

# Send the POST request
response = requests.post(url, json=query)

# Print the response from the server
print(response.json())
