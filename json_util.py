import pandas as pd
import json
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
        "correct_percentage": pd.Series(dtype="float"),  # Float
        "latest_results": pd.Series(dtype="string"),  # String
        "tested_count": pd.Series(dtype="int64"),  # Whole number
    }

    df = pd.DataFrame(data)
    return df


def save_df(df, filename="terms.json", verbose: bool = True):
    """
    Saves a DataFrame as a JSON file with unique_id as the key
    and the remaining columns as dictionary values.
    """
    # Convert DataFrame to a dictionary with unique_id as the key
    data_dict = df.set_index("unique_id").to_dict(orient="index")

    # Save to a JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            data_dict, f, indent=4, default=str
        )  # Ensure dates are converted to strings

    if verbose:
        print(f"Data saved to {filename}")


def load_df(filename="terms.json"):
    """
    Loads a JSON file and converts it back to a Pandas DataFrame.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data_dict = json.load(f)  # Load JSON as a dictionary

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data_dict, orient="index").reset_index()
        df.rename(
            columns={"index": "unique_id"}, inplace=True
        )  # Rename back to unique_id

        # Convert date_last_tested back to datetime
        if "date_last_tested" in df.columns:
            df["date_last_tested"] = pd.to_datetime(
                df["date_last_tested"], errors="coerce"
            )

        print(f"Data loaded from {filename}")
        return df

    except FileNotFoundError:
        print("terms.json not found.")
        return create_term_df()


def load_new_terms(df, directory=".", filename="terms.json"):
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
                        "correct_percentage": 0.0,
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
