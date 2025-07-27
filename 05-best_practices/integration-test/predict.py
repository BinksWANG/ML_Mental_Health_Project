import pickle

from flask import Flask, jsonify, request

with open("mental_health_model_2.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)


def predict(landmine):
    x = dv.transform(landmine)
    preds = model.predict(x)
    return float(preds[0])


app = Flask("landmine-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():

    landmine = request.get_json()

    pred = predict(landmine)

    result = {"landmine_type": pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)