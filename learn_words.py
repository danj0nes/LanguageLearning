import os
import pandas as pd
from json_util import *
from tqdm import tqdm
from datetime import datetime
import keyboard
from enum import Enum

BLANK_RESULTS_STRING = "XXXXXXXXXX"
RECENT_LEARNT_SCORE = 9999
REPEAT_RECENT_LEARNT_SCORE = 10000
REPEAT_LEARNT_SCORE = -1
WEIGHT_1 = 2
WEIGHT_2 = 2
WEIGHT_3 = 1

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"  # Reset to default color

today_date = pd.to_datetime(datetime.today().date())


class ReturnCode(Enum):
    QUIT = 1
    SAVE = 2
    CONTINUE = 3


def calc_learnt_score(df, unique_ids=None):
    """
    Recalculates the learnt_score for each row in the DataFrame.

    If unique_ids are provided, only those rows will be updated.
    If the latest date in 'date_last_tested' is today, the function returns the DataFrame without recalculating.

    Formula:
    learnt_score = WEIGHT_1 * (1 - MIN_MAX(days_since_last_test)) + (WEIGHT_2 * correct_percentage) + WEIGHT_3 * MIN_MAX(tested_count)
    """

    def min_max(min: float, max=float, value=float, inverse: bool = False) -> float:
        if max > min:
            result = (value - min) / (max - min)
            return (1 - result) if inverse else result
        else:
            return 0

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

        # Skip calculation if tested_count == 0
        if row["tested_count"] == 0 and not unique_ids:
            continue

        # Calculate days since last tested
        if pd.notna(row["date_last_tested"]):
            days_since_last_test = (today_date - row["date_last_tested"]).days
        else:
            days_since_last_test = 0

        # Min-max normalize days_since_last_test
        days_since_last_test_normalized = min_max(
            min=min_days, max=max_days, value=days_since_last_test, inverse=True
        )

        # Min-max normalize tested_count
        tested_count_normalized = min_max(
            min=min_tested_count, max=max_tested_count, value=row["tested_count"]
        )

        # Compute new learnt_score
        df.at[i, "learnt_score"] = (
            (WEIGHT_1 * days_since_last_test_normalized)
            + (WEIGHT_2 * row["correct_percentage"])
            + (WEIGHT_3 * tested_count_normalized),
        ) / (WEIGHT_1 + WEIGHT_2 + WEIGHT_3)

    if not unique_ids:
        print("Learnt scores updated.")

    return df


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


def save_result(df, results: list, repeat_incorrect: bool = False):
    recalc_all = False
    recalc_ids = [result[0] for result in results]
    for id, isCorrect, learnt_score in results:
        if isCorrect != None:
            if learnt_score != REPEAT_RECENT_LEARNT_SCORE:
                if (
                    not recalc_all
                    and df.loc[df["unique_id"] == id, "tested_count"]
                    == df["unique_id"].max()
                ):
                    recalc_all = True
                    recalc_ids += df.loc[
                        df["learnt_score"] != RECENT_LEARNT_SCORE
                        and df["learnt_score"] != REPEAT_LEARNT_SCORE,
                        "unique_id",
                    ].values

                df.loc[df["unique_id"] == id, "date_last_tested"] = today_date
                latest_results = df.loc[df["unique_id"] == id, "latest_results"].values[
                    0
                ]
                latest_results = ("O" if isCorrect else "X") + latest_results[:-1]
                df.loc[df["unique_id"] == id, "correct_percentage"] = (
                    latest_results.count("O") / 10
                )
                df.loc[df["unique_id"] == id, "latest_results"] = latest_results
                df.loc[df["unique_id"] == id, "tested_count"] += 1

            if not isCorrect and repeat_incorrect:
                df.loc[df["unique_id"] == id, "learnt_score"] = REPEAT_LEARNT_SCORE
                recalc_ids.remove(id)

    calc_learnt_score(df, recalc_ids)


def get_keypress(
    df: pd.DataFrame,
    recent: list,
    correct: int,
    incorrect: int,
    term: dict,
    showing_term: bool = True,
):
    event = keyboard.read_event(suppress=True)
    id = term["id"]
    if event.event_type == keyboard.KEY_DOWN:  # Detect only key presses (not releases)
        if event.name == "left":
            recent.append((id, False, term["learnt_score"]))
            incorrect += 1
            return df, recent, correct, incorrect, ReturnCode.CONTINUE

        elif event.name == "right":
            recent.append((id, True, term["learnt_score"]))
            correct += 1
            return df, recent, correct, incorrect, ReturnCode.CONTINUE

        elif event.name == "down":
            print("\033[A\033[K", end="")  # Move up and clear the line
            print(f"{term['definition'] if showing_term else term['term']:^32}")
            showing_term = not showing_term
            return get_keypress(df, recent, correct, incorrect, term, showing_term)

        elif event.name == "up":
            if len(recent) > 0:
                df.loc[df["unique_id"] == id, "learnt_score"] = term["learnt_score"]
                df.loc[df["unique_id"] == recent[-1][0], "learnt_score"] = recent[-1][2]
                if recent[-1][1]:
                    correct -= 1
                else:
                    incorrect -= 1
                recent = recent[:-1]
                return df, recent, correct, incorrect, ReturnCode.CONTINUE
            else:
                return get_keypress(df, recent, correct, incorrect, term, showing_term)

        elif event.name == "q":
            recent.append((id, None, term["learnt_score"]))
            return df, recent, correct, incorrect, ReturnCode.QUIT

        elif event.name == "s":
            return df, recent, correct, incorrect, ReturnCode.SAVE

    return get_keypress(df, recent, correct, incorrect, term, showing_term)


def get_top(df) -> dict:
    return {
        "id": df["unique_id"].iloc[0],
        "term": df["term"].iloc[0],
        "definition": df["definition"].iloc[0],
        "learnt_score": df["learnt_score"].iloc[0],
    }


def learn(df, allow_repeats_after: int = 10, repeat_incorrect: bool = True):
    recent = []
    correct = 0
    incorrect = 0
    recent_length = min(df.shape[0], allow_repeats_after)

    print("line 1")
    print("line 2")

    while True:
        df = sort_df(df)
        top_term = get_top(df)

        if (
            repeat_incorrect
            and df.loc[df["unique_id"] == top_term["id"], "learnt_score"].values[0]
            == REPEAT_LEARNT_SCORE
        ):
            df.loc[df["unique_id"] == top_term["id"], "learnt_score"] = (
                REPEAT_RECENT_LEARNT_SCORE
            )
        else:
            df.loc[df["unique_id"] == top_term["id"], "learnt_score"] = (
                RECENT_LEARNT_SCORE
            )

        print("\033[A\033[K", end="")  # Move up and clear the line
        print("\033[A\033[K", end="")

        print(
            f"{RED}{incorrect:>{3}}{RESET}    {BLUE}learnt score: {int(top_term['learnt_score'] * 100)}%{RESET}    {GREEN}{correct}{RESET}"
        )
        print(f"{top_term['term']:^32}")

        df, recent, correct, incorrect, returnCode = get_keypress(
            df, recent, correct, incorrect, top_term
        )

        if returnCode == ReturnCode.CONTINUE:
            if len(recent) > recent_length:
                save_result(df, [recent[0]], repeat_incorrect)
                recent.pop(0)
        elif returnCode == ReturnCode.QUIT:
            print("Exiting...")
            save_result(df, recent)
            save_df(df)
            break
        elif returnCode == ReturnCode.SAVE:
            save_result(df, recent)
            save_df(df)
            recent = []
            correct = 0
            incorrect = 0


terms = load_df()

updated_terms = calc_learnt_score(df=load_new_terms(terms))

learn(df=updated_terms, repeat_incorrect=True)


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
