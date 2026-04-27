"""
Обучение классификатора тематики новостей.
Запускается через ClearML Agent.
"""

import os
import io
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from clearml import Task, Dataset, OutputModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config import cfg

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


task = Task.init(
    project_name=cfg.clearml.project_name,
    task_name=os.environ.get("EXPERIMENT_NAME", "train_logreg"),
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)

dataset_id_path = Path(__file__).parent.parent / "dataset" / "dataset_id.txt"

args = {
    "dataset_id":   os.environ.get(
                        "DATASET_ID",
                        dataset_id_path.read_text().strip()
                        if dataset_id_path.exists() else ""
                    ),
    "max_features": int(os.environ.get("MAX_FEATURES", 30000)),
    "ngram_max":    int(os.environ.get("NGRAM_MAX",    2)),
    "C":            float(os.environ.get("C",          1.0)),
    "max_iter":     int(os.environ.get("MAX_ITER",     300)),
    "solver":       os.environ.get("SOLVER",           "lbfgs"),
    "random_seed":  int(os.environ.get("RANDOM_SEED",  42)),
}

task.connect(args, name="hyperparams")
logger = task.get_logger()

if not args["dataset_id"]:
    raise ValueError("dataset_id не задан!")

# Данные

print(f"Загружаем Dataset {args['dataset_id']}...")
dataset      = Dataset.get(dataset_id=args["dataset_id"])
dataset_path = dataset.get_local_copy()

train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
test_df  = pd.read_csv(os.path.join(dataset_path, "test.csv"))
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

X_train = train_df["text"].tolist()
y_train = train_df["label"].tolist()
X_test  = test_df["text"].tolist()
y_test  = test_df["label"].tolist()

# Модель

print("Векторизация TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=int(args["max_features"]),
    ngram_range=(1, int(args["ngram_max"])),
    sublinear_tf=True,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print("Обучение LogisticRegression...")
clf = LogisticRegression(
    C=float(args["C"]),
    max_iter=int(args["max_iter"]),
    solver=args["solver"],
    random_state=int(args["random_seed"]),
    multi_class="multinomial",
    n_jobs=-1,
)
clf.fit(X_train_vec, y_train)

# Метрики

y_pred      = clf.predict(X_test_vec)
acc         = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy:    {acc:.4f}")
print(f"F1 macro:    {f1_macro:.4f}")
print(f"F1 weighted: {f1_weighted:.4f}")

logger.report_scalar("metrics", "accuracy",    value=acc,         iteration=0)
logger.report_scalar("metrics", "f1_macro",    value=f1_macro,    iteration=0)
logger.report_scalar("metrics", "f1_weighted", value=f1_weighted, iteration=0)

f1_per_class = f1_score(y_test, y_pred, average=None)
for cls, f1_cls in zip(CLASS_NAMES, f1_per_class):
    logger.report_scalar("f1_per_class", cls, value=f1_cls, iteration=0)

# Confusion Matrix

cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
).plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(
    f"Confusion Matrix | C={args['C']} "
    f"max_features={args['max_features']} ngram={args['ngram_max']}"
)
plt.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=120, facecolor="white")
buf.seek(0)

import numpy as np
from PIL import Image
img = Image.open(buf).convert("RGB")
img_array = np.array(img)

logger.report_image(
    title="Confusion Matrix",
    series="test",
    image=img_array,
    iteration=0,
)
plt.close(fig)

# Артефакты

artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

vectorizer_path = artifacts_dir / "vectorizer.pkl"
classifier_path = artifacts_dir / "classifier.pkl"
zip_path        = artifacts_dir / "model_bundle.zip"

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(clf,        classifier_path)

import zipfile
with zipfile.ZipFile(zip_path, "w") as zf:
    zf.write(vectorizer_path, "vectorizer.pkl")
    zf.write(classifier_path, "classifier.pkl")

task.upload_artifact(
    name="vectorizer",
    artifact_object=str(vectorizer_path),
    metadata={"max_features": int(args["max_features"]),
               "ngram_max":    int(args["ngram_max"])},
)
task.upload_artifact(
    name="classifier",
    artifact_object=str(classifier_path),
    metadata={"C": float(args["C"]), "solver": args["solver"]},
)
task.upload_artifact(
    name="model_bundle",
    artifact_object=str(zip_path),
)

output_model = OutputModel(
    task=task,
    name="news_topic_logreg",
    framework="scikit-learn",
    label_enumeration={cls: i for i, cls in enumerate(CLASS_NAMES)},
)
upload_uri = os.environ.get("CLEARML_FILES_HOST", "http://localhost:8081")
output_model.update_weights(
    weights_filename=str(zip_path),
    upload_uri=upload_uri,
    auto_delete_file=False,
)
output_model.update_design(config_dict={
    "vectorizer": {
        "max_features": int(args["max_features"]),
        "ngram_range":  [1, int(args["ngram_max"])],
    },
    "classifier": {
        "C":        float(args["C"]),
        "solver":   args["solver"],
        "max_iter": int(args["max_iter"]),
    },
    "classes":    CLASS_NAMES,
    "accuracy":   acc,
    "f1_macro":   f1_macro,
})

print(f"\n✅ Task завершён | Model ID: {output_model.id}")
task.close()
