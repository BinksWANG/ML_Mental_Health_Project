import lambda_function
import json

def test_prepare_features():
    # Initialize the model service with a test run_id
    test_run_id = "your_test_run_id_here"  # Replace with an actual test run ID
    model_service = lambda_function.ModelService(run_id=test_run_id, model_path="models_mlflow")
    model_service.load_model()  # This will load the model and preprocessor
    
    # Test data - should match the features your model was trained on
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
    # Initialize the model service with a test run_id
    test_run_id = "your_test_run_id_here"  # Replace with an actual test run ID
    model_service = lambda_function.ModelService(run_id=test_run_id, model_path="models_mlflow")
    model_service.load_model()
    
    # Create test event
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