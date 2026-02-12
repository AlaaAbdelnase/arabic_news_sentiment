import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Constants
SVM_MODEL_PATH = 'models/sentiment_svm_model.pkl'
LOGREG_MODEL_PATH = 'models/sentiment_logreg_model.pkl'
RF_MODEL_PATH = 'models/sentiment_rf_model.pkl'
SCALER_PATH = 'sentiment_e5_scaler_4class.pkl'
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-small'
# Model weights based on accuracies
MODEL_WEIGHTS = {
    "svm": 0.7076,
    "logreg": 0.6175,
    "rf": 0.6303,
}
# Labels mapped to your numeric system (-2, -1, 1, 2)
LABELS = {0: "Strongly Negative", 1: "Negative", 2: "Positive", 3: "Strongly Positive"}

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Use POST /predict with JSON { 'text': '...' }"
    }), 200

# Load models once on startup
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    with open(SVM_MODEL_PATH, 'rb') as f:
        svm_model = pickle.load(f)
    with open(LOGREG_MODEL_PATH, 'rb') as f:
        logreg_model = pickle.load(f)
    with open(RF_MODEL_PATH, 'rb') as f:
        rf_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Models loaded successfully")
except Exception as e:
    print(f"Loading error: {e}")
    embedding_model = None
    svm_model = None
    logreg_model = None
    rf_model = None
    scaler = None

def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def _get_model_probs(model, feat_scaled: np.ndarray):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat_scaled)[0]
        classes = model.classes_.tolist()
        return classes, probs
    if hasattr(model, "decision_function"):
        scores = model.decision_function(feat_scaled)[0]
        scores = np.atleast_1d(scores)
        probs = _softmax(scores)
        classes = model.classes_.tolist()
        return classes, probs
    pred = int(model.predict(feat_scaled)[0])
    return [pred], np.array([1.0])

@app.route('/predict', methods=['POST'])
def predict():
    if embedding_model is None or svm_model is None or logreg_model is None or rf_model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    data = request.get_json(silent=True) or {}
    raw_text = data.get('text')
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "Field 'text' must be a non-empty string"}), 400

    try:
        # 1. Embedding (E5 requires 'query: ' prefix for best results)
        feat = embedding_model.encode([f"query: {raw_text}"])

        # 2. Scaling
        feat_scaled = scaler.transform(feat)

        # 3. Prediction & Weighted Voting
        model_outputs = {}
        weighted_scores = {}

        for name, model in (
            ("svm", svm_model),
            ("logreg", logreg_model),
            ("rf", rf_model),
        ):
            classes, probs = _get_model_probs(model, feat_scaled)
            weight = MODEL_WEIGHTS.get(name, 1.0)

            # Normalize if probabilities not summing to 1 (defensive)
            probs = np.array(probs, dtype=float)
            if probs.sum() > 0:
                probs = probs / probs.sum()

            model_outputs[name] = {
                "probabilities": {
                    LABELS.get(int(cls), str(int(cls))): float(prob)
                    for cls, prob in zip(classes, probs)
                },
                "predicted_class": int(classes[int(np.argmax(probs))]),
                "predicted_label": LABELS.get(int(classes[int(np.argmax(probs))]), str(int(classes[int(np.argmax(probs))]))),
                "confidence": float(np.max(probs)),
                "weight": float(weight),
            }

            for cls, prob in zip(classes, probs):
                weighted_scores[int(cls)] = weighted_scores.get(int(cls), 0.0) + weight * float(prob)

        # Normalize weighted scores
        total_weighted = sum(weighted_scores.values())
        if total_weighted > 0:
            for cls in list(weighted_scores.keys()):
                weighted_scores[cls] = weighted_scores[cls] / total_weighted

        best_class = max(weighted_scores, key=weighted_scores.get)

        response = {
            "label": LABELS.get(best_class, str(best_class)),
            "class_index": int(best_class),
            "probabilities": {
                LABELS.get(cls, str(cls)): float(prob)
                for cls, prob in weighted_scores.items()
            },
            "confidence": float(weighted_scores.get(best_class, 0.0)),
            "model_outputs": model_outputs,
            "weights": MODEL_WEIGHTS,
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)