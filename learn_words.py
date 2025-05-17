import pandas as pd
from csv_util import *
from util import validate_int_bound
from tqdm import tqdm
from datetime import datetime
import keyboard
from enum import Enum

WEIGHT_DAYS_SINCE = 1
WEIGHT_CORRECT = 1
WEIGHT_TESTED = 1

TESTED_MAX_CAP = 15
TESTED_CAP_WEIGHTING = 0.9
LATEST_RESULTS_LENGTH = 10

DESIRED_TERMS_TYPES = ["verbe", "mot", "nom", "adjectif", "phrase", "other"]
ALLOW_REPEATS_AFTER = 15
MIN_LIST_NUMBER = None
MAX_LIST_NUMBER = None

# terminal colour codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
RESET = "\033[0m"  # Reset to default color

today_date = pd.to_datetime(datetime.today().date())


class ReturnCode(Enum):
    QUIT = 1
    TERMINATE = 2
    SAVE = 3
    CONTINUE = 4
    REVERSE = 5


def calc_learnt_score(df, unique_ids=None, verbose: bool = False) -> pd.DataFrame:
    """
    Recalculates the learnt_score for each row in the DataFrame.

    If unique_ids are provided, only those rows will be updated.
    Terms with tested_count == 0 will not be updated

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
    else:
        rows_to_update = df.index

    # Get the max for days_since_last_test and tested_count columns
    max_days = (today_date - df["date_last_tested"].min()).days
    max_tested_count = df["tested_count"].max()

    # Update learnt_score for each term
    for i in tqdm(rows_to_update, desc="Updating learnt scores", disable=not verbose):
        row = df.loc[i]

        # Skip calculation if tested_count == 0
        if row["tested_count"] == 0:
            continue

        # Min-max normalize days_since_last_test
        if pd.notna(row["date_last_tested"]):
            days_since_last_test = (today_date - row["date_last_tested"]).days

            days_since_last_test_normalized = min_max(
                min=0, max=max_days, value=days_since_last_test, inverse=True
            )
        else:
            days_since_last_test_normalized = 0

        # Calculate percentage correct
        if row["latest_results"] != BLANK_RESULTS_STRING:
            correct_percentage = row["latest_results"].count("O") / max(
                LATEST_RESULTS_LENGTH, len(row["latest_results"])
            )
        else:
            correct_percentage = 0

        # Piecewise weighted Min-max normalize tested_count
        lower_piece = (
            min(row["tested_count"], TESTED_MAX_CAP) / TESTED_MAX_CAP
        ) * TESTED_CAP_WEIGHTING

        upper_piece = min_max(
            min=TESTED_MAX_CAP,
            max=max(max_tested_count, TESTED_MAX_CAP),
            value=max(row["tested_count"], TESTED_MAX_CAP),
        ) * (1 - TESTED_CAP_WEIGHTING)

        tested_count_normalized = lower_piece + upper_piece

        # Compute new learnt_score
        df.at[i, "learnt_score"] = round(
            (
                (WEIGHT_DAYS_SINCE * days_since_last_test_normalized)
                + (WEIGHT_CORRECT * correct_percentage)
                + (WEIGHT_TESTED * tested_count_normalized)
            )
            / (WEIGHT_DAYS_SINCE + WEIGHT_CORRECT + WEIGHT_TESTED),
            4,
        )

    if verbose:
        print("Learnt scores updated.")

    return sort_df(df)


def sort_df(df: pd.DataFrame):
    """
    Shuffles dataframe then sorts via learnt_score.
    Therefore, the of order of terms with same learnt score will be random.
    """
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    return shuffled_df.sort_values(
        by=["learnt_score"], ascending=[True], ignore_index=True
    )


def save_result(
    df,
    recent: list,
    repeat_incorrect_ids: list,
    terminating: bool = False,
    recent_gap: int = 0,
) -> pd.DataFrame:

    recalc_all = False

    # if terminating, save results of all incorrect terms waiting to be repeated
    if terminating:
        for repeat_incorrect_id, _ in repeat_incorrect_ids:
            recent.append((repeat_incorrect_id, False, True))

    if not recent:  # if recent empty or None
        return df

    for index, (id, isCorrect, repeat_incorrect) in enumerate(recent):
        if isCorrect or terminating:
            # if max tested_count will be broken then recalc ~all learnt scores
            if not recalc_all:
                tested_count = df.loc[df["unique_id"] == id, "tested_count"].values[0]
                if (
                    tested_count == df["tested_count"].max()
                    and tested_count >= TESTED_MAX_CAP
                ):
                    recalc_all = True

            # update date_last_test
            df.loc[df["unique_id"] == id, "date_last_tested"] = today_date

            # update lastest_results
            latest_results = df.loc[df["unique_id"] == id, "latest_results"].values[0]
            if latest_results != BLANK_RESULTS_STRING:
                latest_results = ("X" if repeat_incorrect else "O") + (
                    latest_results
                    if len(latest_results) < LATEST_RESULTS_LENGTH
                    else latest_results[:-1]
                )
            else:
                latest_results = "X" if repeat_incorrect else "O"
            df.loc[df["unique_id"] == id, "latest_results"] = latest_results

            # update tested_count
            df.loc[df["unique_id"] == id, "tested_count"] += 1
        else:
            # ensure when saving, incorrect terms are tested again with sufficient repeat after delay
            repeat_incorrect_ids.append((id, index + recent_gap))

    return calc_learnt_score(
        df, None if recalc_all else [result[0] for result in recent]
    )


def get_top(
    df: pd.DataFrame,
    recent: list,
    repeat_incorrect_ids: list,
    future_terms: list,
    reversing: bool = False,
    quitting: bool = False,
):

    temp_df = df.set_index("unique_id", drop=False)  # Optimise lookup

    # filtering df
    temp_df = temp_df[
        temp_df["term_type"].isin(DESIRED_TERMS_TYPES)
        & (temp_df["list_number"] >= MIN_LIST_NUMBER)
        & (temp_df["list_number"] <= MAX_LIST_NUMBER)
    ]

    def term_data(term_id: str, repeat: bool) -> dict:
        row = temp_df.loc[term_id]
        return {
            "id": term_id,
            "term": row["term"],
            "definition": row["definition"],
            "learnt_score": row["learnt_score"],
            "repeat_incorrect": repeat,
        }

    data = None

    # 1. Pick from future terms
    if future_terms:
        id, repeat_incorrect = (
            future_terms.pop(0) if not reversing else recent.pop()[:3:2]
        )
        data = term_data(id, repeat_incorrect)

    else:
        # 2. Pick from repeat incorrect
        if repeat_incorrect_ids:
            if repeat_incorrect_ids[0][1] == 0 or quitting:
                data = term_data(repeat_incorrect_ids.pop(0)[0], True)

        # 3. Pick from filtered df
        if data is None and not quitting:
            avoid_ids = {entry[0] for entry in recent}
            for id, _ in temp_df.iterrows():
                if id not in avoid_ids:
                    data = term_data(id, False)
                    break

        # reduce repeat incorrect delay if given a brand new term
        if data is not None:
            repeat_incorrect_ids = [
                (x, y - 1 if y > 0 else y) for x, y in repeat_incorrect_ids
            ]

    return recent, repeat_incorrect_ids, future_terms, data


def get_keypress(
    recent: list,
    correct: int,
    incorrect: int,
    term: dict,
):
    showing_term = True
    id = term["id"]
    while True:
        event = keyboard.read_event(suppress=True)

        # Detect only key releases (not pressess/holds)
        if event.event_type != keyboard.KEY_UP:
            continue

        action = None

        if event.name == "left":
            recent.append((id, False, term["repeat_incorrect"]))
            if not term["repeat_incorrect"]:
                incorrect += 1
            action = ReturnCode.CONTINUE

        elif event.name == "right":
            recent.append((id, True, term["repeat_incorrect"]))
            if not term["repeat_incorrect"]:
                correct += 1
            action = ReturnCode.CONTINUE

        elif event.name == "down":
            text = term["definition"] if showing_term else term["term"]
            print("\033[A\033[K", end="")  # Move up and clear the line
            print(f"{YELLOW if term['repeat_incorrect'] else ''}{text:^32}{RESET}")
            showing_term = not showing_term
            continue  # Skip return

        elif event.name == "up" and recent:
            if not recent[-1][2]:  # if last tested term was not repeat incorrect
                if recent[-1][1]:  # if last tested term was correct
                    correct -= 1
                else:
                    incorrect -= 1
            action = ReturnCode.REVERSE

        elif event.name == "q":
            action = ReturnCode.QUIT

        elif event.name == "t":
            action = ReturnCode.TERMINATE

        elif event.name == "s":
            action = ReturnCode.SAVE

        if action:
            return recent, correct, incorrect, action


def learn(df: pd.DataFrame, file: str):
    recent = []
    future_terms = []
    repeat_incorrect_ids = []
    correct = 0
    incorrect = 0

    recent_length = min(df.shape[0] - 1, ALLOW_REPEATS_AFTER)
    reversing = False
    quitting = False

    print("line 1")
    print("line 2")

    while True:
        # get next term to learn
        recent, repeat_incorrect_ids, future_terms, top_term = get_top(
            df=df,
            recent=recent,
            repeat_incorrect_ids=repeat_incorrect_ids,
            future_terms=future_terms,
            reversing=reversing,
            quitting=quitting,
        )

        if not top_term:
            if not quitting:
                print("No terms to learn...")
            break

        reversing = False

        print("\033[A\033[K", end="")  # Move up and clear the line
        print("\033[A\033[K", end="")
        print(
            f"{RED}{incorrect:>{3}}{RESET}{'':4}{BLUE}learnt score: {int(top_term['learnt_score'] * 100)}%{RESET}{'':4}"
            + f"{GREEN}{correct}{RESET}{'':4}{PURPLE}{'quitting' if quitting else ''}{RESET}"
        )
        print(
            f"{YELLOW if top_term['repeat_incorrect'] else ''}{top_term['term']:^32}{RESET}"
        )

        recent, correct, incorrect, returnCode = get_keypress(
            recent=recent, correct=correct, incorrect=incorrect, term=top_term
        )

        if returnCode == ReturnCode.CONTINUE:
            if len(recent) > recent_length or (
                quitting and len(repeat_incorrect_ids) == 0 and len(future_terms) == 0
            ):
                df = save_result(df, [recent[0]], repeat_incorrect_ids)
                recent.pop(0)

        elif returnCode == ReturnCode.REVERSE:
            # ensure than term on screen is chosen again
            future_terms.insert(0, (top_term["id"], top_term["repeat_incorrect"]))
            reversing = True

        elif returnCode == ReturnCode.SAVE:
            df = save_result(
                df,
                recent,
                repeat_incorrect_ids,
                recent_gap=recent_length - len(recent),
            )
            save_df(df=df, file=file, verbose=False)

            # ensure than term on screen is chosen again
            future_terms.insert(0, (top_term["id"], top_term["repeat_incorrect"]))

            recent = []
            correct = 0
            incorrect = 0

        elif returnCode == ReturnCode.QUIT:
            quitting = True

            # adds all incorrect terms to repeat_incorrect
            df = save_result(df, recent, repeat_incorrect_ids)
            save_df(df=df, file=file, verbose=False)

            # keep only repeat incorrect terms of furture_terms
            future_terms = [term for term in future_terms if term[1]]

            # ensure than term on screen is chosen again
            if top_term["repeat_incorrect"]:
                future_terms.insert(0, (top_term["id"], True))

            recent = []
            correct = 0
            incorrect = 0

        elif returnCode == ReturnCode.TERMINATE:
            print("Exiting...")
            # ensure that term on screen if gotten wrong is saved as gotten wrong
            if top_term["repeat_incorrect"]:
                repeat_incorrect_ids.insert(0, (top_term["id"], 0))

            # add all repeat incorrect terms in future_terms to repeat_incorrect_ids
            repeat_incorrect_ids.extend(
                [(term[0], 0) for term in future_terms if term[1]]
            )
            break

    df = save_result(df, recent, repeat_incorrect_ids, terminating=True)
    save_df(df=df, file=file)


def main():
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terms.csv")

    terms = load_df(file=file)

    updated_terms = calc_learnt_score(
        df=load_new_lists(df=terms, file=file), verbose=True
    )

    global MIN_LIST_NUMBER, MAX_LIST_NUMBER
    MIN_LIST_NUMBER = validate_int_bound(
        MIN_LIST_NUMBER, updated_terms["list_number"].min(), "lower_bound"
    )
    MAX_LIST_NUMBER = validate_int_bound(
        MAX_LIST_NUMBER, updated_terms["list_number"].max(), "upper_bound"
    )

    learn(df=updated_terms, file=file)


if __name__ == "__main__":
    main()
