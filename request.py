import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'AGE': 25, 'CUS_Month_Income': 50000, 'YEARS_WITH_US': 12})

print(r.json())