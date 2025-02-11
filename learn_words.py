import os
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
import keyboard

BLANK_RESULTS_STRING = "XXXXXXXXXX"
WEIGHT_1 = 2
WEIGHT_2 = 2
WEIGHT_3 = 1

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"  # Reset to default color

today_date = pd.to_datetime(datetime.today().date())

recent = []
correct = 0
incorrect = 0


def calc_learnt_score(df, unique_ids=None):
    """
    Recalculates the learnt_score for each row in the DataFrame.

    If unique_ids are provided, only those rows will be updated.
    If the latest date in 'date_last_tested' is today, the function returns the DataFrame without recalculating.

    Formula:
    learnt_score = WEIGHT_1 * (1 - MIN_MAX(days_since_last_test)) + (WEIGHT_2 * correct_percentage) + WEIGHT_3 * MIN_MAX(tested_count)
    """

    # Select rows to update
    if unique_ids:
        rows_to_update = df[df["unique_id"].isin(unique_ids)].index
        use_tqdm = False  # No progress bar for specific IDs
    else:
        # Check if the latest test date is today
        if (
            df["date_last_tested"].dropna().empty
            or df["date_last_tested"].max().date() == today_date
        ):
            print("Latest test date is today. No recalculations needed.")
            return df

        rows_to_update = df.index
        use_tqdm = True  # Use progress bar when updating all rows

    # Get the min and max for days_since_last_test and tested_count columns
    min_days = (today_date - df["date_last_tested"].max()).days
    max_days = (today_date - df["date_last_tested"].min()).days

    min_tested_count = df["tested_count"].min()
    max_tested_count = df["tested_count"].max()

    # Update learnt_score
    for i in tqdm(rows_to_update, desc="Updating learnt scores", disable=not use_tqdm):
        row = df.loc[i]

        # # Skip calculation if tested_count == 0
        # if row["tested_count"] == 0:
        #     continue

        # Calculate days since last tested
        if pd.notna(row["date_last_tested"]):
            days_since_last_test = (today_date - row["date_last_tested"]).days
        else:
            days_since_last_test = 0

        # Min-max normalize days_since_last_test
        if max_days > min_days:
            days_since_last_test_normalized = 1 - (
                (days_since_last_test - min_days) / (max_days - min_days)
            )
        else:
            days_since_last_test_normalized = 0

        # Min-max normalize tested_count
        if max_tested_count > min_tested_count:
            tested_count_normalized = (row["tested_count"] - min_tested_count) / (
                max_tested_count - min_tested_count
            )
        else:
            tested_count_normalized = 0

        # Compute new learnt_score
        df.at[i, "learnt_score"] = (
            (WEIGHT_1 * days_since_last_test_normalized)
            + (WEIGHT_2 * row["correct_percentage"])
            + (WEIGHT_3 * tested_count_normalized)
        )

    print("Learnt scores updated.")
    return df


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


def sort_df(df):
    """
    Sorts the DataFrame by increasing learnt_score.
    """
    return df.sort_values(
        by=["learnt_score", "unique_id"], ascending=[True, True], ignore_index=True
    )


def save_result(df, results: list):
    ids = []
    print(results)
    for id, isCorrect, learnt_score in results:
        if isCorrect != None:
            df.loc[df["unique_id"] == id, "tested_count"] += 1
            df.loc[df["unique_id"] == id, "date_last_tested"] = today_date
            latest_results = df.loc[df["unique_id"] == id, "latest_results"].values[0]
            latest_results = ("O" if isCorrect else "X") + latest_results[:-1]
            df.loc[df["unique_id"] == id, "latest_results"] = latest_results
            df.loc[df["unique_id"] == id, "correct_percentage"] = (
                latest_results.count("O") / 10
            )
        ids.append(id)

    calc_learnt_score(df, ids)


def get_keypress(df, id, term, definition, learnt_score, showing_term: bool = True):
    global recent
    global correct
    global incorrect
    event = keyboard.read_event(suppress=True)
    if event.event_type == keyboard.KEY_DOWN:  # Detect only key presses (not releases)
        if event.name == "left":
            recent.append((id, False, learnt_score))
            incorrect += 1
            return df, True
        elif event.name == "right":
            recent.append((id, True, learnt_score))
            correct += 1
            return df, True
        elif event.name == "down":
            print("\033[A\033[K", end="")  # Move up and clear the line
            print(
                f"{RED}{incorrect}{RESET}  {definition if showing_term else term}  {GREEN}{correct}{RESET}  learnt score: {learnt_score}"
            )
            showing_term = not showing_term
            return get_keypress(df, id, term, definition, learnt_score, showing_term)
        elif event.name == "up":
            if len(recent) > 0:
                df.loc[df["unique_id"] == id, "learnt_score"] = learnt_score
                df.loc[df["unique_id"] == recent[-1][0], "learnt_score"] = recent[-1][2]
                if recent[-1][1]:
                    correct -= 1
                else:
                    incorrect -= 1
                recent = recent[:-1]
            else:
                return get_keypress(
                    df, id, term, definition, learnt_score, showing_term
                )
            return df, True
        elif event.name == "q":
            recent.append((id, None, learnt_score))
            print("Exiting...")
            save_result(df, recent)
            save_df(df)
            return df, False
    return get_keypress(df, id, term, definition, learnt_score, showing_term)


def learn(df):
    while True:
        df = sort_df(df)
        id = df["unique_id"].iloc[0]
        term = df["term"].iloc[0]
        definition = df["definition"].iloc[0]
        learnt_score = df["learnt_score"].iloc[0]
        df.loc[df["unique_id"] == id, "learnt_score"] = 99999

        print("\033[A\033[K", end="")  # Move up and clear the line
        print(
            f"{RED}{incorrect}{RESET}  {term}  {GREEN}{correct}{RESET}  learnt score: {learnt_score}"
        )

        df, notQuit = get_keypress(df, id, term, definition, learnt_score)

        if notQuit:
            if len(recent) >= 10:
                save_result(df, [recent[0]])
                recent.pop(0)
        else:
            break


terms = load_df()

updated_terms = calc_learnt_score(df=load_new_terms(terms))

learn(df=updated_terms)


"""
gets top phrase from df with lowest learnt score
 - set learnt score as inf

user then gets it right or wrong

current phrases tested (array length 10)
 - saves df unique id and result
 - add most current to end

 if current phrases > 10 
 - save result of top in current phrases
 - sort df


 must save when quit


df struction

id      learnt score     term    definition      date last tested    correct percentage  tested count


TODO

sort json by unique id
sort df by learnscore then unique id
"""
