"""
Загрузка датасета AG News в ClearML.
"""

import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from clearml import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config import cfg


def prepare_and_upload() -> str:
    print("Загружаем AG News из HuggingFace...")

    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    train_size  = 4000
    test_size   = 1000
    total       = train_size + test_size

    raw = load_dataset("ag_news", split=f"train[:{total}]")

    df = pd.DataFrame({
        "text":       raw["text"],
        "label":      raw["label"],
        "label_name": [label_names[l] for l in raw["label"]],
    })

    train_df = df.iloc[:train_size]
    test_df  = df.iloc[train_size:]

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(df["label_name"].value_counts())

    output_dir = Path(__file__).parent
    train_path = output_dir / "train.csv"
    test_path  = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    dataset = Dataset.create(
        dataset_name=cfg.clearml.dataset_name,
        dataset_project=cfg.clearml.project_name,
        dataset_version=cfg.clearml.dataset_version,
        description="AG News: 4 класса тематики новостей",
    )

    dataset.add_files(str(train_path))
    dataset.add_files(str(test_path))

    dataset.get_logger().report_table(
        title="Train sample",
        series="head",
        table_plot=train_df.head(10),
    )

    dataset.upload()
    dataset.finalize()

    dataset_id = dataset.id
    print(f"\n✅ Dataset ID: {dataset_id}")

    id_path = output_dir / "dataset_id.txt"
    id_path.write_text(dataset_id)
    print(f"   ID сохранён в {id_path}")

    return dataset_id


if __name__ == "__main__":
    prepare_and_upload()
