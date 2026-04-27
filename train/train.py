"""
Скрипт обучения классификатора тематики новостей.
Запускается через ClearML Agent в очереди 'students'.

Гиперпараметры задаются через Task.connect или при клонировании.
"""

import os
import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clearml import Task, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import joblib


task = Task.init(
    project_name="news_topic_classification",
    task_name="train_logreg",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)

args = {
    "dataset_id":   "",
    "max_features": 30000,       # размер словаря TF-IDF
    "ngram_max":    2,           # (1,ngram_max) — n-граммы
    "C":            1.0,         # регуляризация LogReg
    "max_iter":     300,
    "solver":       "lbfgs",
    "random_seed":  42,
}

task.connect(args, name="hyperparams")
if not args["dataset_id"]:
    try:
        with open(os.path.join(os.path.dirname(__file__),
                               "../dataset/dataset_id.txt")) as f:
            args["dataset_id"] = f.read().strip()
    except FileNotFoundError:
        raise ValueError(
            "Укажите dataset_id в гиперпараметрах или в файле dataset/dataset_id.txt"
        )

logger = task.get_logger()

print(f"Загружаем Dataset {args['dataset_id']} из ClearML...")
dataset = Dataset.get(dataset_id=args["dataset_id"])
dataset_path = dataset.get_local_copy()

train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
test_df  = pd.read_csv(os.path.join(dataset_path, "test.csv"))

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

X_train = train_df["text"].tolist()
y_train = train_df["label"].tolist()
X_test  = test_df["text"].tolist()
y_test  = test_df["label"].tolist()

# TF-IDF + LogReg
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
y_pred = clf.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy:    {acc:.4f}")
print(f"F1 macro:    {f1_macro:.4f}")
print(f"F1 weighted: {f1_weighted:.4f}")

logger.report_scalar("metrics", "accuracy",    value=acc,         iteration=0)
logger.report_scalar("metrics", "f1_macro",    value=f1_macro,    iteration=0)
logger.report_scalar("metrics", "f1_weighted", value=f1_weighted, iteration=0)

for i, cls in enumerate(CLASS_NAMES):
    f1_cls = f1_score(y_test, y_pred, average=None)[i]
    logger.report_scalar("f1_per_class", cls, value=f1_cls, iteration=0)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(
    f"Confusion Matrix\n"
    f"C={args['C']} | max_features={args['max_features']} | ngram_max={args['ngram_max']}"
)
plt.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=120)
buf.seek(0)
logger.report_image(
    title="Confusion Matrix",
    series="test",
    image=plt.imread(buf),
    iteration=0,
)
plt.close(fig)

# Сохранение артефактов

os.makedirs("artifacts", exist_ok=True)
joblib.dump(vectorizer, "artifacts/vectorizer.pkl")
joblib.dump(clf,        "artifacts/classifier.pkl")

task.upload_artifact(
    name="vectorizer",
    artifact_object="artifacts/vectorizer.pkl",
    metadata={
        "max_features": int(args["max_features"]),
        "ngram_max":    int(args["ngram_max"]),
    }
)
task.upload_artifact(
    name="classifier",
    artifact_object="artifacts/classifier.pkl",
    metadata={
        "C":       float(args["C"]),
        "solver":  args["solver"],
        "classes": CLASS_NAMES,
    }
)

# Output Model
from clearml import OutputModel
import zipfile

# Упакуем оба файла в один zip — удобнее для serving
zip_path = "artifacts/model_bundle.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    zf.write("artifacts/vectorizer.pkl",  "vectorizer.pkl")
    zf.write("artifacts/classifier.pkl",  "classifier.pkl")

output_model = OutputModel(
    task=task,
    name="news_topic_logreg",
    framework="scikit-learn",
    label_enumeration={cls: i for i, cls in enumerate(CLASS_NAMES)},
)
output_model.update_weights(
    weights_filename=zip_path,
    auto_delete_file=False,
)
output_model.update_design(config_dict={
    "vectorizer": {
        "max_features": int(args["max_features"]),
        "ngram_range":  [1, int(args["ngram_max"])],
    },
    "classifier": {
        "C":       float(args["C"]),
        "solver":  args["solver"],
        "max_iter": int(args["max_iter"]),
    },
    "classes": CLASS_NAMES,
    "accuracy": acc,
    "f1_macro": f1_macro,
})

print(f"\nTask завершён.")
print(f"   Model ID: {output_model.id}")
task.close()
