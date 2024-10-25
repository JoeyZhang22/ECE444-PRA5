from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

print("Loading Model")
loaded_model = None
with open("basic_classifier.pkl", "rb") as fid:
    loaded_model = pickle.load(fid)

vectorizer = None
with open("count_vectorizer.pkl", "rb") as vd:
    vectorizer = pickle.load(vd)


@application.route("/")
def index():
    return "Your Flask App Works! V1.0"


@application.route("/test_prediction", methods=["POST"])
def prediction():
    data = request.get_json()
    text = data.get("text", "")
    if len(text) == 0:
        return jsonify({"error": "Null Text"}), 400
    # From lab handout
    prediction = loaded_model.predict(vectorizer.transform([text]))[0]
    return jsonify({"text": text, "prediction": "fake" if prediction == 1 else "real"})122


if __name__ == "__main__":
    application.run(port=5000, debug=True)
