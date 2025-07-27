# pylint: disable=duplicate-code

import json

import requests
from deepdiff import DeepDiff

with open('event.json', 'rt', encoding='utf-8') as f_in:
    event = json.load(f_in)


url = "http://0.0.0.0:9696/predict"
actual_response = requests.post(url, json=event).json()
print('actual response:')

print(json.dumps(actual_response, indent=2))

expected_response = {
    'predictions': [
        {
            'model': 'mental_health_model_2',
            'version': 'Test123',
            'prediction': {
                "Age": 47,
                "Sleep_Hours": 7.7,
                "Work_Hours": 31,
                "Physical_Activity_Hours": 10
            },
        }
    ]
}


diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff