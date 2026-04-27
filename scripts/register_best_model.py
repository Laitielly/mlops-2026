"""
Находит лучший эксперимент и регистрирует модель.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config import cfg
from clearml import Task, Model


def get_best_task():
    tasks = Task.get_tasks(
        project_name=cfg.clearml.project_name,
        task_filter={
            "status": ["completed"],
            "type":   ["training"],
        }
    )

    if not tasks:
        raise RuntimeError("Нет завершённых задач обучения")

    best_task = None
    best_f1   = -1.0

    for t in tasks:
        try:
            f1 = t.get_last_scalar_metrics()["metrics"]["f1_macro"]["last"]
        except (KeyError, TypeError):
            continue

        print(f"Task '{t.name}' | f1_macro={f1:.4f}")

        if f1 > best_f1:
            best_f1   = f1
            best_task = t

    if best_task is None:
        raise RuntimeError("Нет задач с метрикой f1_macro")

    print(f"\n🏆 Лучшая задача: '{best_task.name}' (f1_macro={best_f1:.4f})")
    return best_task, best_f1


def register(best_task: Task, best_f1: float) -> str:
    models = best_task.get_models()["output"]
    if not models:
        raise RuntimeError(f"У задачи {best_task.id} нет output-моделей")

    model: Model = models[0]
    print(f"Модель: {model.name} | ID: {model.id}")

    scalars = best_task.get_last_scalar_metrics()
    acc = scalars.get("metrics", {}).get("accuracy", {}).get("last", 0.0)

    model.publish()
    model.tags = ["production", "logreg", "ag_news"]

    print(f"\n✅ Модель зарегистрирована")
    print(f"   Model ID : {model.id}")
    print(f"   Tags     : {model.tags}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 macro : {best_f1:.4f}")

    root = Path(__file__).parent.parent
    (root / "best_model_id.txt").write_text(model.id)
    (root / "best_task_id.txt").write_text(best_task.id)

    return model.id


if __name__ == "__main__":
    best_task, best_f1 = get_best_task()
    register(best_task, best_f1)
