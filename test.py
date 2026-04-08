import requests

url = "https://human-ai-optimizer.hf.space"

response = requests.post(f"{url}/reset", verify=False)  # temporary
print(response.text)