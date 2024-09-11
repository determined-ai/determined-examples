import datasets
import pandas as pd


def add_length_column(dataset) -> pd.DataFrame:
    df = dataset.to_pandas()
    df["total_length"] = 0
    for column_name in ["instruction", "input", "response"]:
        num_words = df[column_name].astype(str).str.split().apply(len)
        df["total_length"] += num_words

    return df


def filter_by_total_length(df, difficulty, number_of_samples):
    if difficulty == "easy":
        return df[df["total_length"].between(10, 100)].iloc[:number_of_samples]
    elif difficulty == "medium":
        return df[df["total_length"].between(101, 200)].iloc[:number_of_samples]
    elif difficulty == "hard":
        return df[df["total_length"].between(201, 800)].iloc[:number_of_samples]


def get_dataset_subset_name(difficulty: str) -> str:
    return f"text-to-sql-v1-{difficulty}"


def create_and_save_datasets(
    df, difficulty, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
):
    seed = 123
    # remove total_length column because we don't need it anymore
    df = df.drop(columns=["total_length"])
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)

    # split into training and "the rest"
    train_valtest = dataset.train_test_split(train_size=train_ratio, seed=seed)

    # split "the rest" into validation and testing
    val_test = train_valtest["test"].train_test_split(
        test_size=test_ratio / (test_ratio + val_ratio), seed=seed
    )

    dataset = datasets.DatasetDict(
        {
            "train": train_valtest["train"],
            "valid": val_test["train"],
            "test": val_test["test"],
        }
    )
    dataset_name = get_dataset_subset_name(difficulty)
    dataset.save_to_disk(dataset_name)
    return dataset


def load_dataset(difficulty):
    return datasets.load_from_disk(get_dataset_subset_name(difficulty))


def load_or_create_dataset(difficulty, num_samples=10000):
    try:
        return load_dataset(difficulty)
    except FileNotFoundError:
        dataset = datasets.load_dataset("Clinton/Text-to-sql-v1")
        dataset = dataset["train"]
        dataset = dataset.remove_columns(["text", "source"])
        df = add_length_column(dataset)
        df = filter_by_total_length(df, difficulty, num_samples)
        return create_and_save_datasets(df, difficulty)