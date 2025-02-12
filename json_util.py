import pandas as pd
import json


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


def save_df(df, filename="terms.json"):
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
