"""
Находит лучший эксперимент по f1_macro и публикует модель в Model Registry.
"""

from clearml import Task, Model


PROJECT = "news_topic_classification"


def get_best_task() -> Task:
    """Возвращает задачу с наибольшим f1_macro."""
    tasks = Task.get_tasks(
        project_name=PROJECT,
        task_filter={
            "status": ["completed"],
            "type":   ["training"],
        }
    )

    best_task   = None
    best_f1     = -1.0

    for t in tasks:
        scalars = t.get_last_scalar_metrics()
        try:
            f1 = scalars["metrics"]["f1_macro"]["last"]
        except (KeyError, TypeError):
            continue

        print(f"Task '{t.name}' | f1_macro={f1:.4f}")

        if f1 > best_f1:
            best_f1   = f1
            best_task = t

    if best_task is None:
        raise RuntimeError("Нет завершённых задач с метрикой f1_macro")

    print(f"\n🏆 Лучшая задача: '{best_task.name}' (f1_macro={best_f1:.4f})")
    return best_task, best_f1


def register(best_task: Task, best_f1: float):
    """Публикует модель из задачи в Model Registry."""

    models = best_task.get_models()["output"]
    if not models:
        raise RuntimeError(f"У задачи {best_task.id} нет output-моделей")

    source_model: Model = models[0]
    print(f"Модель-источник: {source_model.name} | ID: {source_model.id}")

    scalars = best_task.get_last_scalar_metrics()
    acc = scalars.get("metrics", {}).get("accuracy", {}).get("last", 0.0)
    source_model.publish()
    source_model.update_tags(["production", "logreg", "ag_news"])

    config = source_model.get_model_design() or {}
    config.update({
        "registry_note": "Best model by f1_macro",
        "accuracy":      acc,
        "f1_macro":      best_f1,
        "source_task":   best_task.id,
    })
    source_model.update_design(config_dict=config)

    print(f"\n✅ Модель зарегистрирована в Registry")
    print(f"   Model ID : {source_model.id}")
    print(f"   Tags     : {source_model.tags}")

    with open("best_model_id.txt", "w") as f:
        f.write(source_model.id)

    return source_model.id


if __name__ == "__main__":
    best_task, best_f1 = get_best_task()
    model_id = register(best_task, best_f1)
