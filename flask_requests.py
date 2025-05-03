import requests
import json

# Define the URL of the Flask API endpoint
url = "http://127.0.0.1:5000/predict"

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Define the data payload
data = {
    "text": "I hate using Hugging Face models!"
}

# Send the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))

# Check the response
if response.status_code == 200:
    # Parse and print the JSON response
    result = response.json()
    print("Prediction Result:", result)
else:
    print(f"Request failed with status code {response.status_code}")
    print("Response:", response.text)
