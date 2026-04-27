"""
Загрузка датасета AG News (классификация тематики новостей).

Классы:
    0 - World
    1 - Sports
    2 - Business
    3 - Sci/Tech
"""

import pandas as pd
from datasets import load_dataset
from clearml import Dataset


def prepare_and_upload():
    print("Загружаем AG News из HuggingFace...")
    raw = load_dataset("ag_news", split="train[:5000]")

    df = pd.DataFrame({
        "text": raw["text"],
        "label": raw["label"],
        "label_name": [
            ["World", "Sports", "Business", "Sci/Tech"][l]
            for l in raw["label"]
        ]
    })

    train_df = df.iloc[:4000]
    test_df  = df.iloc[4000:]

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv",   index=False)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(df["label_name"].value_counts())

    dataset = Dataset.create(
        dataset_name="ag_news_topic",
        dataset_project="news_topic_classification",
        dataset_version="1.0",
        description="AG News: 4 классa тематики новостей (5 000 примеров)",
    )

    dataset.add_files("train.csv")
    dataset.add_files("test.csv")

    dataset.get_logger().report_table(
        title="Train sample",
        series="head",
        table_plot=train_df.head(10),
    )

    dataset.upload()
    dataset.finalize()

    print(f"\nDataset ID: {dataset.id}")
    print("   Виден в ClearML UI → Datasets → ag_news_topic")
    return dataset.id


if __name__ == "__main__":
    dataset_id = prepare_and_upload()
    import os
    os.makedirs("dataset", exist_ok=True)
    with open("dataset/dataset_id.txt", "w") as f:
        f.write(dataset_id)
