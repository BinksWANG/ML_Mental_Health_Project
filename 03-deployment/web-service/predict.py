import pickle

from flask import Flask, request, jsonify

with open('mental_health_model_2.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(data):
    features = {
        "Age": data["Age"],
        "Sleep_Hours": data["Sleep_Hours"],
        "Work_Hours": data["Work_Hours"],
        "Physical_Activity_Hours": data["Physical_Activity_Hours"]
    }
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

app = Flask('mental-health-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()

    features = prepare_features(data)
    pred = predict(features)

    result = {
        'mental-health-issue': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)