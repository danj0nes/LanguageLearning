import pandas as pd
from json_util import *
from tqdm import tqdm
from datetime import datetime
import keyboard
from enum import Enum

WEIGHT_DAYS_SINCE = 2
WEIGHT_CORRECT = 2
WEIGHT_TESTED = 1

# terminal colour codes
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
    learnt_score = WEIGHT_DAYS_SINCE * (1 - MIN_MAX(days_since_last_test)) + (WEIGHT_CORRECT * correct_percentage) + WEIGHT_TESTED * MIN_MAX(tested_count) / (WEIGHT_DAYS_SINCE + WEIGHT_CORRECT + WEIGHT_TESTED)
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
            (WEIGHT_DAYS_SINCE * days_since_last_test_normalized)
            + (WEIGHT_CORRECT * row["correct_percentage"])
            + (WEIGHT_TESTED * tested_count_normalized)
        ) / (WEIGHT_DAYS_SINCE + WEIGHT_CORRECT + WEIGHT_TESTED)

    if not unique_ids:
        print("Learnt scores updated.")

    return df


def sort_df(df):
    """
    Sorts the DataFrame by increasing learnt_score.
    """
    return df.sort_values(
        by=["learnt_score", "unique_id"], ascending=[True, True], ignore_index=True
    )


def save_result(df, results: list, repeat_incorrect_ids: list):
    recalc_all = False

    for id, isCorrect, repeat_incorrect in results:
        if isCorrect != None:  # isCorrect it None when q is pressed on term
            # if term repeated because already gotten wrong then stats shouldn't update again
            if not repeat_incorrect:
                # if max tested_count is broken then recalc ~all learnt scores
                if (
                    not recalc_all
                    and df.loc[df["unique_id"] == id, "tested_count"].values[0]
                    == df["unique_id"].max()
                ):
                    recalc_all = True

                # update stats for term with result
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
                # end of update stats

            if not isCorrect:
                repeat_incorrect_ids.append(id)

    calc_learnt_score(df, None if recalc_all else [result[0] for result in results])


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
            recent.append((id, False, term["repeat_incorrect"]))
            incorrect += 1
            return df, recent, correct, incorrect, ReturnCode.CONTINUE

        elif event.name == "right":
            recent.append((id, True, term["repeat_incorrect"]))
            correct += 1
            return df, recent, correct, incorrect, ReturnCode.CONTINUE

        elif event.name == "down":
            print("\033[A\033[K", end="")  # Move up and clear the line
            if not term["repeat_incorrect"]:
                print(f"{term['definition'] if showing_term else term['term']:^32}")
            else:
                print(
                    f"{YELLOW}{term['definition'] if showing_term else term['term']:^32}{RESET}"
                )
            showing_term = not showing_term
            return get_keypress(df, recent, correct, incorrect, term, showing_term)

        elif event.name == "up":
            if len(recent) > 0:
                if recent[-1][1]:  # if got prev term correct
                    correct -= 1
                else:
                    incorrect -= 1
                recent = recent[:-1]
                return df, recent, correct, incorrect, ReturnCode.CONTINUE
            else:
                return get_keypress(df, recent, correct, incorrect, term, showing_term)

        elif event.name == "q":
            recent.append((id, None, term["repeat_incorrect"]))
            return df, recent, correct, incorrect, ReturnCode.QUIT

        elif event.name == "s":
            return df, recent, correct, incorrect, ReturnCode.SAVE

    return get_keypress(df, recent, correct, incorrect, term, showing_term)


def get_top(df: pd.DataFrame, recent: list, repeat_incorrect_ids: list) -> dict:
    if len(repeat_incorrect_ids) > 0:
        id = repeat_incorrect_ids[0]
        repeat_incorrect_ids.remove(id)
        return {
            "id": id,
            "term": df.loc[df["unique_id"] == id, "term"].values[0],
            "definition": df.loc[df["unique_id"] == id, "definition"].values[0],
            "learnt_score": df.loc[df["unique_id"] == id, "learnt_score"].values[0],
            "repeat_incorrect": True,
        }

    avoid_ids = set(result[0] for result in recent)
    for _, row in df.iterrows():
        if row["unique_id"] not in avoid_ids:
            return {
                "id": row["unique_id"],
                "term": row["term"],
                "definition": row["definition"],
                "learnt_score": row["learnt_score"],
                "repeat_incorrect": False,
            }

    return None  # Return None if no valid row is found


def learn(df, allow_repeats_after: int = 10):
    recent = []
    repeat_incorrect_ids = []
    correct = 0
    incorrect = 0
    recent_length = min(df.shape[0], allow_repeats_after)

    print("line 1")
    print("line 2")

    while True:
        df = sort_df(df)
        top_term = get_top(df, recent, repeat_incorrect_ids)

        print("\033[A\033[K", end="")  # Move up and clear the line
        print("\033[A\033[K", end="")

        print(
            f"{RED}{incorrect:>{3}}{RESET}    {BLUE}learnt score: {int(top_term['learnt_score'] * 100)}%{RESET}    {GREEN}{correct}{RESET}"
        )
        if top_term["repeat_incorrect"]:
            print(f"{YELLOW}{top_term['term']:^32}{RESET}")
        else:
            print(f"{top_term['term']:^32}")

        df, recent, correct, incorrect, returnCode = get_keypress(
            df, recent, correct, incorrect, top_term
        )

        if returnCode == ReturnCode.CONTINUE:
            if len(recent) > recent_length:
                save_result(df, [recent[0]], repeat_incorrect_ids)
                recent.pop(0)
        elif returnCode == ReturnCode.QUIT:
            print("Exiting...")
            save_result(df, recent, repeat_incorrect_ids)
            save_df(df)
            break
        elif returnCode == ReturnCode.SAVE:
            save_result(df, recent, repeat_incorrect_ids)
            save_df(df, verbose=False)
            recent = []
            correct = 0
            incorrect = 0


terms = load_df()

updated_terms = calc_learnt_score(df=load_new_terms(terms))

learn(df=updated_terms)


"""

df struction

id      learnt score     term    definition      date last tested    correct percentage  tested count


TODO
save what happends to repeat incorrect
"""
