"""
Inference сервер.
Загружает модель из ClearML Registry.
Готов к деплою на сервак через gunicorn.
"""

import os
import sys
import zipfile
import logging
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from clearml import Model, Task

sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config import cfg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


app = Flask(__name__)

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

_vectorizer = None
_classifier = None
_model_id   = None


def load_model_from_registry():
    """Загружает модель из ClearML Model Registry."""
    global _vectorizer, _classifier, _model_id

    root     = Path(__file__).parent.parent
    model_id = (root / "best_model_id.txt").read_text().strip()
    task_id  = (root / "best_task_id.txt").read_text().strip()

    log.info(f"Загружаем модель {model_id} из ClearML Registry...")

    model      = Model(model_id=model_id)
    model_path = model.get_local_copy()
    log.info(f"Модель скачана: {model_path}")

    extract_dir = root / "artifacts" / "unpacked"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if os.path.isdir(model_path):
        vec_path = Path(model_path) / "vectorizer.pkl"
        clf_path = Path(model_path) / "classifier.pkl"
    elif str(model_path).endswith(".zip"):
        with zipfile.ZipFile(model_path, "r") as zf:
            zf.extractall(extract_dir)
        vec_path = extract_dir / "vectorizer.pkl"
        clf_path = extract_dir / "classifier.pkl"
    elif str(model_path).endswith("vectorizer.pkl"):
        base     = Path(model_path).parent
        vec_path = Path(model_path)
        clf_path = base / "classifier.pkl"
    else:
        vec_path = Path(model_path)
        clf_path = None

    if clf_path is None or not clf_path.exists():
        log.warning("Classifier не найден рядом, берём из артефактов задачи...")
        task     = Task.get_task(task_id=task_id)
        clf_path = Path(task.artifacts["classifier"].get_local_copy())

    if not vec_path.exists():
        log.warning("Vectorizer не найден рядом, берём из артефактов задачи...")
        task     = Task.get_task(task_id=task_id)
        vec_path = Path(task.artifacts["vectorizer"].get_local_copy())

    _vectorizer = joblib.load(vec_path)
    _classifier = joblib.load(clf_path)
    _model_id   = model_id

    log.info("✅ Модель загружена успешно!")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "model_id": _model_id,
        "loaded":   _vectorizer is not None,
    })


@app.route("/serve/news-topic", methods=["POST"])
def predict():
    if _vectorizer is None or _classifier is None:
        return jsonify({"error": "Модель не загружена"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Тело запроса должно быть JSON"}), 400
    if "text" not in data:
        return jsonify({"error": "Поле 'text' обязательно"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Текст не может быть пустым"}), 400

    vec      = _vectorizer.transform([text])
    proba    = _classifier.predict_proba(vec)[0]
    pred_idx = int(np.argmax(proba))

    return jsonify({
        "prediction":    pred_idx,
        "label":         CLASS_NAMES[pred_idx],
        "confidence":    round(float(proba[pred_idx]), 4),
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(CLASS_NAMES, proba)
        },
    })


if __name__ == "__main__":
    load_model_from_registry()
    host = os.getenv("SERVING_HOST", cfg.serving.host)
    port = int(os.getenv("SERVING_PORT", cfg.serving.port))
    log.info(f"🚀 Сервер на http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
