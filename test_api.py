import requests, json, sys
url = 'http://127.0.0.1:5000/api/detect/image'
# send sample name as form data
data = {'sample': 'sample_helmet_violation.jpg'}
resp = requests.post(url, data=data)
print('Status:', resp.status_code)
print('Response:', resp.text)
