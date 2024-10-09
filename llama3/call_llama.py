import requests
import json

def call_llama(data):
    try:
        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)

        # Check if the response was successful
        if response.status_code == 200:
            # Parse the JSON response
            json_data = response.json()

            # Check if the 'response' key is in the JSON
            if "response" in json_data:
                # print(f"response = {json_data["response"]}")
                # Print the generated JSON data nicely formatted
                print(f"output = {json.dumps(json.loads(json_data["response"]), indent=2)}")
            else:
                # If 'response' key is not found, print the entire JSON response
                print("Key 'response' not found in the response:")
                print(json.dumps(json_data, indent=2))
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as e:
        # Handle connection errors or timeouts
        print(f"An error occurred: {e}")