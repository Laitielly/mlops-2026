"""
ClearML Serving препроцессинг-скрипт.
Размещается в ClearML Serving как 'preprocess' модуль.

Схема работы:
    request JSON {"text": "..."} 
    → preprocess → model.predict 
    → postprocess → response JSON
"""

import os
import zipfile
import numpy as np
import joblib

from clearml.serving.model_request_handler import BasePreprocessRequest


CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

_vectorizer = None
_classifier = None


def _load_model(local_model_dir: str):
    """Распаковываем zip-бандл и загружаем артефакты."""
    global _vectorizer, _classifier

    zip_files = [f for f in os.listdir(local_model_dir) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError(
            f"Не найден model_bundle.zip в {local_model_dir}"
        )

    zip_path = os.path.join(local_model_dir, zip_files[0])
    extract_dir = os.path.join(local_model_dir, "unpacked")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    _vectorizer = joblib.load(os.path.join(extract_dir, "vectorizer.pkl"))
    _classifier = joblib.load(os.path.join(extract_dir, "classifier.pkl"))
    print("✅ Модель загружена из ClearML Registry")


class Preprocess(BasePreprocessRequest):

    def preprocess(
        self,
        body: dict,
        state: dict,
        collect_custom_statistics_fn=None,
    ):
        """Извлекаем текст из запроса."""
        text = body.get("text", "")
        if not text:
            raise ValueError("Поле 'text' обязательно")
        return [text]

    def process(
        self,
        data,
        state: dict,
        collect_custom_statistics_fn=None,
    ):
        """
        Вызывается после preprocess.
        data — это то, что вернул preprocess.
        """
        global _vectorizer, _classifier

        if _vectorizer is None or _classifier is None:
            _load_model(state.get("model_dir", "."))

        vec = _vectorizer.transform(data)
        proba = _classifier.predict_proba(vec)[0]
        pred_idx = int(np.argmax(proba))

        return {
            "prediction": pred_idx,
            "label":      CLASS_NAMES[pred_idx],
            "confidence": float(proba[pred_idx]),
            "probabilities": {
                cls: float(p) for cls, p in zip(CLASS_NAMES, proba)
            },
        }

    def postprocess(
        self,
        data,
        state: dict,
        collect_custom_statistics_fn=None,
    ):
        """Возвращаем результат как есть."""
        return data
