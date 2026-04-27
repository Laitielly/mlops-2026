"""
Запускает эксперименты из конфига.
"""

import subprocess
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config import cfg


def main():
    dataset_id_path = Path(__file__).parent.parent / "dataset" / "dataset_id.txt"

    if not dataset_id_path.exists():
        raise FileNotFoundError(
            "dataset/dataset_id.txt не найден. "
            "Сначала запусти: python dataset/upload_dataset.py"
        )

    dataset_id = dataset_id_path.read_text().strip()
    print(f"Dataset ID: {dataset_id}\n")

    for exp in cfg.training.experiments:
        print(f"🚀 Запускаем: {exp.name}")

        env = os.environ.copy()
        env.update({
            "EXPERIMENT_NAME": exp.name,
            "DATASET_ID":      dataset_id,
            "MAX_FEATURES":    str(exp.max_features),
            "NGRAM_MAX":       str(exp.ngram_max),
            "C":               str(exp.C),
            "MAX_ITER":        str(exp.max_iter),
            "SOLVER":          exp.solver,
            "RANDOM_SEED":     str(exp.random_seed),
        })

        result = subprocess.run(
            [sys.executable, "train/train.py"],
            env=env,
        )

        status = "✅" if result.returncode == 0 else "❌"
        print(f"{status} {exp.name} завершён (code={result.returncode})\n")


if __name__ == "__main__":
    main()
