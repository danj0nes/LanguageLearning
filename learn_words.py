import pandas as pd
from csv_util import *
from tqdm import tqdm
from datetime import datetime
import keyboard
from enum import Enum

WEIGHT_DAYS_SINCE = 1
WEIGHT_CORRECT = 1
WEIGHT_TESTED = 1

TESTED_MAX_CAP = 15
TESTED_CAP_WEIGHTING = 0.9

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
    REVERSE = 4


def calc_learnt_score(df, unique_ids=None) -> pd.DataFrame:
    """
    Recalculates the learnt_score for each row in the DataFrame.

    If unique_ids are provided, only those rows will be updated.
    If the latest date in 'date_last_tested' is today, the function returns the DataFrame without recalculating.

    Formula:
    learnt_score = WEIGHT_DAYS_SINCE * (1 - MIN_MAX(days_since_last_test)) + (WEIGHT_CORRECT * correct_percentage) + WEIGHT_TESTED * PIECEWISE_WEIGHTED_MIN_MAX(tested_count) / (WEIGHT_DAYS_SINCE + WEIGHT_CORRECT + WEIGHT_TESTED)
    """

    def min_max(min: float, max=float, value=float, inverse: bool = False) -> float:
        if max > min:
            result = (value - min) / (max - min)
            return (1 - result) if inverse else result
        else:
            return 1 if inverse else 0

    # Select rows to update
    if unique_ids:
        rows_to_update = df[df["unique_id"].isin(unique_ids)].index
        use_tqdm = False  # No progress bar for specific IDs
    else:
        rows_to_update = df.index
        use_tqdm = True  # Use progress bar when updating all rows

    # Get the min and max for days_since_last_test and tested_count columns
    max_days = (today_date - df["date_last_tested"].min()).days

    max_tested_count = df["tested_count"].max()

    # Update learnt_score
    for i in tqdm(rows_to_update, desc="Updating learnt scores", disable=not use_tqdm):
        row = df.loc[i]

        # Skip calculation if tested_count == 0
        if row["tested_count"] == 0:
            continue

        # Calculate days since last tested
        if pd.notna(row["date_last_tested"]):
            days_since_last_test = (today_date - row["date_last_tested"]).days
        else:
            days_since_last_test = None

        # Calculate percentage correct
        correct_percentage = row["latest_results"].count("O") / len(
            row["latest_results"]
        )

        # Min-max normalize days_since_last_test
        if days_since_last_test != None:
            days_since_last_test_normalized = min_max(
                min=0, max=max_days, value=days_since_last_test, inverse=True
            )
        else:
            days_since_last_test_normalized = 0

        # Piecewise weighted Min-max normalize tested_count
        tested_count_normalized = (
            min(row["tested_count"], TESTED_MAX_CAP) / TESTED_MAX_CAP
        ) * TESTED_CAP_WEIGHTING + min_max(
            min=TESTED_MAX_CAP,
            max=max(max_tested_count, TESTED_MAX_CAP),
            value=max(row["tested_count"], TESTED_MAX_CAP),
        ) * (
            1 - TESTED_CAP_WEIGHTING
        )
        # Compute new learnt_score
        df.at[i, "learnt_score"] = (
            (WEIGHT_DAYS_SINCE * days_since_last_test_normalized)
            + (WEIGHT_CORRECT * correct_percentage)
            + (WEIGHT_TESTED * tested_count_normalized)
        ) / (WEIGHT_DAYS_SINCE + WEIGHT_CORRECT + WEIGHT_TESTED)

    if not unique_ids:
        print("Learnt scores updated.")

    return sort_df(df)


def sort_df(df):
    return df.sort_values(
        by=["learnt_score", "unique_id"], ascending=[True, True], ignore_index=True
    )


def save_result(df, recent: list, repeat_incorrect_ids: list) -> pd.DataFrame:
    recalc_all = False

    if not recent:
        return df

    for id, isCorrect, repeat_incorrect in recent:
        if isCorrect:
            # if max tested_count is broken then recalc ~all learnt scores
            if (
                not recalc_all
                and df.loc[df["unique_id"] == id, "tested_count"].values[0]
                == df["unique_id"].max()
            ):
                recalc_all = True

            # update stats for term with result
            df.loc[df["unique_id"] == id, "date_last_tested"] = today_date
            latest_results = df.loc[df["unique_id"] == id, "latest_results"].values[0]
            latest_results = ("X" if repeat_incorrect else "O") + (
                latest_results if len(latest_results) < 10 else latest_results[:-1]
            )
            df.loc[df["unique_id"] == id, "latest_results"] = latest_results
            df.loc[df["unique_id"] == id, "tested_count"] += 1
            # end of update stats
        else:
            repeat_incorrect_ids.append(id)

    return calc_learnt_score(
        df, None if recalc_all else [result[0] for result in recent]
    )


def get_keypress(
    recent: list,
    correct: int,
    incorrect: int,
    term: dict,
):
    showing_term = True
    while True:
        event = keyboard.read_event(suppress=True)
        # Detect only key releases (not pressess/holds)
        # this is here to
        if event.event_type != keyboard.KEY_UP:
            continue

        id = term["id"]
        action = None

        if event.name == "left":
            recent.append((id, False, term["repeat_incorrect"]))
            incorrect += 1
            action = ReturnCode.CONTINUE

        elif event.name == "right":
            recent.append((id, True, term["repeat_incorrect"]))
            correct += 1
            action = ReturnCode.CONTINUE

        elif event.name == "down":
            print("\033[A\033[K", end="")  # Move up and clear the line
            text = term["definition"] if showing_term else term["term"]
            print(f"{YELLOW if term['repeat_incorrect'] else ''}{text:^32}{RESET}")
            showing_term = not showing_term
            continue  # Skip return

        elif event.name == "up" and recent:
            if recent[-1][1]:  # if got prev term correct
                correct -= 1
            else:
                incorrect -= 1
            action = ReturnCode.REVERSE

        elif event.name == "q":
            action = ReturnCode.QUIT

        elif event.name == "s":
            action = ReturnCode.SAVE

        if action:
            return recent, correct, incorrect, action


def get_top(
    df: pd.DataFrame,
    recent: list,
    repeat_incorrect_ids: list,
    future_terms: list,
    reversing: bool,
):

    df = df.set_index("unique_id", drop=False)  # Optimize lookup
    term_data = lambda id, repeat: {
        "id": id,
        "term": df.at[id, "term"],
        "definition": df.at[id, "definition"],
        "learnt_score": df.at[id, "learnt_score"],
        "repeat_incorrect": repeat,
    }

    if future_terms:
        id, repeat_incorrect = (
            future_terms.pop() if not reversing else recent.pop()[:3:2]
        )
        return recent, future_terms, term_data(id, repeat_incorrect)

    if repeat_incorrect_ids:
        return recent, future_terms, term_data(repeat_incorrect_ids.pop(0), True)

    avoid_ids = {entry[0] for entry in recent}
    for id, _ in df.iterrows():
        if id not in avoid_ids:
            return recent, future_terms, term_data(id, False)

    return recent, None, None  # No valid row found


def learn(df, allow_repeats_after: int = 9):
    recent = []
    future_terms = []
    repeat_incorrect_ids = []
    correct = 0
    incorrect = 0

    recent_length = min(df.shape[0], allow_repeats_after)
    reversing = False

    print("line 1")
    print("line 2")

    while True:
        # get next term to learn
        recent, future_terms, top_term = get_top(
            df, recent, repeat_incorrect_ids, future_terms, reversing
        )

        reversing = False

        if not top_term:
            print("No terms to learn...")
            break

        print("\033[A\033[K", end="")  # Move up and clear the line
        print("\033[A\033[K", end="")
        print(
            f"{RED}{incorrect:>{3}}{RESET}{'':4}{BLUE}learnt score: {int(top_term['learnt_score'] * 100)}%{RESET}{'':4}{GREEN}{correct}{RESET}"
        )
        print(
            f"{YELLOW if top_term['repeat_incorrect'] else ''}{top_term['term']:^32}{RESET}"
        )

        recent, correct, incorrect, returnCode = get_keypress(
            recent, correct, incorrect, top_term
        )

        if returnCode == ReturnCode.CONTINUE:
            if len(recent) > recent_length:
                df = save_result(df, [recent[0]], repeat_incorrect_ids)
                recent.pop(0)

        elif returnCode == ReturnCode.REVERSE:
            future_terms.append((top_term["id"], top_term["repeat_incorrect"]))
            reversing = True

        elif returnCode == ReturnCode.QUIT:
            print("Exiting...")
            break

        elif returnCode == ReturnCode.SAVE:
            df = save_result(df, recent, repeat_incorrect_ids)
            save_df(df, verbose=False)
            future_terms.append((top_term["id"], top_term["repeat_incorrect"]))
            recent = []
            correct = 0
            incorrect = 0

    df = save_result(df, recent, repeat_incorrect_ids)
    save_df(df)


terms = load_df()

updated_terms = calc_learnt_score(df=load_new_terms(terms))

learn(df=updated_terms)
