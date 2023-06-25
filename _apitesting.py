import requests

url = 'http://127.0.0.1:8000/predict'  # Correct URL
data = {'text': 'This is a positive review.'}  # Replace with your input data

response = requests.post(url, json=data)
print(response.json())  # Assuming the response is JSON