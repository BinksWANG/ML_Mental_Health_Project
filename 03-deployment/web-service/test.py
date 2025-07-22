import requests

## categorical = ['Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 'Severity', 'Consultation_History', 'Stress_Level']
## numerical = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours']

data = {
    "Age": 47,
    "Sleep_Hours": 7.7,
    "Work_Hours": 31,
    "Physical_Activity_Hours": 10
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=data)
print(response.json())