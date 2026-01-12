"""Simple Flask service for breast cancer model prediction.

Endpoints:
- GET /health -> returns {"status": "ok"}
- POST /predict -> JSON body with either:
    {"sample_index": 0}
  or
    {"features": [v1, v2, ..., v30]}

Returns JSON: {"prediction": 0/1, "prediction_name": "malignant"/"benign", "probability": [p0, p1]}
"""
from flask import Flask, request, jsonify, Response, url_for, send_from_directory
import os
import joblib
from sklearn.datasets import load_breast_cancer

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")

app = Flask(__name__)

print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found at {MODEL_PATH}. Run training first or set MODEL_PATH env var.")

saved = joblib.load(MODEL_PATH)
if isinstance(saved, dict) and "model" in saved:
    clf = saved["model"]
else:
    clf = saved

data = load_breast_cancer()
feature_names = list(data.feature_names)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def index():
        html = f"""
        <html>
            <head><title>Breast Cancer Model API</title></head>
            <body>
                <h2>Breast Cancer Model API</h2>
                <ul>
                    <li><a href="{url_for('health')}">/health</a> - service health</li>
                    <li>/predict (POST) - JSON body: {{'sample_index':0}} or {{'features':[...30 values...]}}</li>
                </ul>
                <p>Example curl (sample 0):</p>
                <pre>curl -X POST -H "Content-Type: application/json" -d '{{"sample_index":0}}' http://127.0.0.1:5000/predict</pre>
            </body>
        </html>
        """
        return Response(html, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    if body is None:
        return jsonify({"error": "Request must be JSON"}), 400

    if "sample_index" in body:
        idx = int(body["sample_index"])
        if idx < 0 or idx >= len(data.data):
            return jsonify({"error": f"sample_index out of range 0..{len(data.data)-1}"}), 400
        x = data.data[idx]
    elif "features" in body:
        vals = body["features"]
        if not isinstance(vals, (list, tuple)):
            return jsonify({"error": "features must be a list"}), 400
        if len(vals) != len(feature_names):
            return jsonify({"error": f"Expected {len(feature_names)} features"}), 400
        try:
            x = [float(v) for v in vals]
        except Exception:
            return jsonify({"error": "All feature values must be numeric"}), 400
    else:
        return jsonify({"error": "Provide 'sample_index' or 'features' in JSON body"}), 400

    pred = int(clf.predict([x])[0])
    proba = clf.predict_proba([x])[0].tolist() if hasattr(clf, "predict_proba") else None
    return jsonify({"prediction": pred, "prediction_name": data.target_names[pred], "probability": proba})


@app.route('/report', methods=['GET'])
def report_page():
    """Serve the HTML report if present under ./reports/report.html"""
    reports_dir = os.path.join(os.getcwd(), 'reports')
    report_file = os.path.join(reports_dir, 'report.html')
    if not os.path.exists(report_file):
        return jsonify({"error": "Report not found. Run: python3 breast_cancer_predict.py report --out-dir reports"}), 404
    return send_from_directory(reports_dir, 'report.html')


@app.route('/reports/<path:filename>', methods=['GET'])
def reports_static(filename):
    """Serve static files from the reports directory (images)"""
    reports_dir = os.path.join(os.getcwd(), 'reports')
    if not os.path.exists(os.path.join(reports_dir, filename)):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(reports_dir, filename)


if __name__ == "__main__":
    # For development only. Use gunicorn/uwsgi for production.
    app.run(host="0.0.0.0", port=5000)
