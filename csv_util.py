import pandas as pd
import os
from tqdm import tqdm


BLANK_RESULTS_STRING = "XXXXXXXXXX"


def create_term_df():
    """
    Creates a Pandas DataFrame with specified columns and data types.
    """
    data = {
        "unique_id": pd.Series(dtype="int64"),  # Whole number
        "learnt_score": pd.Series(dtype="float"),  # Decimal number
        "term": pd.Series(dtype="string"),  # String
        "definition": pd.Series(dtype="string"),  # String
        "date_last_tested": pd.Series(dtype="datetime64[ns]"),  # Date
        "latest_results": pd.Series(dtype="string"),  # String
        "tested_count": pd.Series(dtype="int64"),  # Whole number
    }

    df = pd.DataFrame(data)
    return df


def save_df(df: pd.DataFrame, filename: str = "terms.csv", verbose: bool = True):
    """
    Saves a DataFrame as a CSV file.

    index=False stops the index of the df being written in file.
    """
    df.to_csv(filename, index=False)

    if verbose:
        print(f"Data saved to {filename}")


def load_df(filename="terms.csv"):
    """
    Loads a CSV file and converts it back to a Pandas DataFrame.
    """
    if not os.path.exists(filename):
        print("terms.json not found.")
        return create_term_df()

    df = pd.read_csv(
        filename,
        parse_dates=["date_last_tested"],
        dtype={
            "unique_id": int,
            "learnt_score": float,
            "term": str,
            "definition": str,
            "latest_results": str,
            "tested_count": int,
        },
    )
    # df["date_last_tested"] = pd.to_datetime(df["date_last_tested"], errors="coerce")
    print(f"Data loaded from {filename}")
    return df


def load_new_terms(df, directory=".", filename="terms.csv"):
    """
    Walks through the given directory, extracts term-definition pairs from .txt files,
    and adds any new terms to the DataFrame.
    """
    print("Searching for new terms.")
    new_entries = []

    highest_id = df["unique_id"].max() if not df.empty else 0

    existing_terms = set(df["term"])  # Get existing terms for quick lookup
    txt_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith(".txt")
    ]

    # Use tqdm to track progress over files
    for file_path in tqdm(txt_files, desc="Processing text files"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Extract term-definition pairs (every two lines)
        for i in range(0, len(lines) - 1, 2):
            term = lines[i].strip()
            definition = lines[i + 1].strip()

            # Only add new terms
            if term not in existing_terms:
                new_entries.append(
                    {
                        "unique_id": highest_id + 1,
                        "learnt_score": 0.0,
                        "term": term,
                        "definition": definition,
                        "date_last_tested": pd.NaT,
                        "latest_results": BLANK_RESULTS_STRING,
                        "tested_count": 0,
                    }
                )
                highest_id += 1
                existing_terms.add(term)  # Avoid duplicates within this run

    # Append new data and save if there are new entries
    if new_entries:
        df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
        save_df(df)  # Save the updated DataFrame
        print(f"Added {len(new_entries)} new terms to {filename}.")
    else:
        print("No new terms found.")

    return df
