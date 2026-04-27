"""
Отправляет два эксперимента с разными гиперпараметрами в очередь 'students'.
"""

from clearml import Task

DATASET_ID = open("dataset/dataset_id.txt").read().strip()

EXPERIMENTS = [
    {
        "name": "logreg_baseline",
        "params": {
            "dataset_id":   DATASET_ID,
            "max_features": 20000,
            "ngram_max":    1,
            "C":            1.0,
            "max_iter":     300,
            "solver":       "lbfgs",
            "random_seed":  42,
        }
    },
    {
        "name": "logreg_bigram_highC",
        "params": {
            "dataset_id":   DATASET_ID,
            "max_features": 50000,
            "ngram_max":    2,
            "C":            5.0,
            "max_iter":     500,
            "solver":       "lbfgs",
            "random_seed":  42,
        }
    },
]


for exp in EXPERIMENTS:
    task = Task.create(
        project_name="news_topic_classification",
        task_name=exp["name"],
        script="train/train.py",
        add_task_init_call=True,
    )

    task.set_parameters_as_dict({"hyperparams": exp["params"]})
    task.execute_remotely(queue_name="students", clone=False, exit_process=False)

    print(f"✅ Эксперимент '{exp['name']}' отправлен в очередь 'students'")
    print(f"   Task ID: {task.id}")
