import model
import json

def test_base64_decode():
    base64_input = "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ=="
    model.base64_decode(base64_input) 
    actual_features = model.base64_decode(data)
    expected_features = {
        "data":{
            "Age": 47,
            "Sleep_Hours": 7.7,
            "Work_Hours": 31,
            "Physical_Activity_Hours": 10
        }
    }
    assert actual_features == expected_features



def test_prepare_features():
    model_service = model.ModelService(None)

    data = {
        "Age": 47,
        "Sleep_Hours": 7.7,
        "Work_Hours": 31,
        "Physical_Activity_Hours": 10
    }
    
    # Prepare features
    features = model_service.prepare_features(data)
    
    # Make prediction
    prediction = model_service.predict(features)
    
    # Basic assertions
    assert features is not None
    assert prediction is not None
    assert isinstance(prediction[0], float)  # Assuming regression output

def test_lambda_handler():
    model_service = model.ModelService(None)

    test_event = {
        "Records": [
            {
                "body": json.dumps({
                    "Age": 47,
                    "Sleep_Hours": 7.7,
                    "Work_Hours": 31,
                    "Physical_Activity_Hours": 10
                })
            }
        ]
    }
    
    # Call lambda_handler
    result = model_service.lambda_handler(test_event)
    
    # Verify the response structure
    assert 'predictions' in result
    assert len(result['predictions']) == 1
    prediction = result['predictions'][0]
    assert 'prediction' in prediction
    assert 'model' in prediction
    assert 'run_id' in prediction
    assert isinstance(prediction['prediction'], float)