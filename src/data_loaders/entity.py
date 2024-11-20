import os
import pandas as pd


def _prepare_entity_dataframe(path):
    df = pd.read_csv(path)

    df["label"] = df["label"].apply(lambda x: [x])
    df = df.groupby("text").agg("sum").reset_index()
    df = df[["text", "label"]]

    df["label"] = df["label"].apply(lambda x: sorted(list(set(x))))

    return df


def _load_entity(validation_file, test_file):
    base_dir = "./data"
    entity_dir = os.path.join(base_dir, "entity")

    task_files = {
        "validation": os.path.join(entity_dir, validation_file)
        if validation_file
        else None,
        "test": os.path.join(entity_dir, test_file),
    }
    
    # get val and test set
    validation_df = (
        _prepare_entity_dataframe(task_files["validation"])
        if task_files["validation"]
        else None
    )
    test_df = _prepare_entity_dataframe(task_files["test"])
    
    return validation_df, test_df, None, None, None
