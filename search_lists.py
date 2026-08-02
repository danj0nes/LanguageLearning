from pathlib import Path
import pandas as pd

QUIT_TOKEN = ":q"

EXCLUDED_DIRS = {"Archived Term Lists", "pyenv"}

CSV_CACHE = {}
TXT_CACHE = {}


def iter_files(root_dir):
    """Yield all non-excluded files."""
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue

        yield path


def load_files():
    """Load all TXT and CSV files into memory."""
    script_dir = Path(__file__).resolve().parent

    csv_count = 0
    txt_count = 0

    for path in iter_files(script_dir):
        suffix = path.suffix.lower()

        if suffix == ".csv":
            try:
                df = pd.read_csv(path)

                if {"TERM", "DEFINITION"}.issubset(df.columns):
                    CSV_CACHE[path.name] = (
                        df[["TERM", "DEFINITION"]].fillna("").astype(str)
                    )

                    csv_count += 1

            except Exception as e:
                print(f"Failed to load {path.name}: {e}")

        elif suffix == ".txt":
            try:
                entries = []

                with path.open("r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f]

                for i in range(0, len(lines) - 1, 2):
                    entries.append((lines[i], lines[i + 1]))

                TXT_CACHE[path.name] = entries
                txt_count += 1

            except Exception as e:
                print(f"Failed to load {path.name}: {e}")

    return csv_count, txt_count


def search_txt(entries, search_term_lower):
    """Search cached TXT entries."""
    results = []

    for french, english in entries:
        if search_term_lower in french.lower() or search_term_lower in english.lower():
            results.append((french, english))

    return results


def search_csv(df, search_term):
    """Search cached CSV dataframe."""
    mask = df["TERM"].str.contains(search_term, case=False, regex=False) | df[
        "DEFINITION"
    ].str.contains(
        search_term,
        case=False,
        regex=False,
    )

    matches = df.loc[mask, ["TERM", "DEFINITION"]]

    return list(matches.itertuples(index=False, name=None))


def search_all(search_term):
    """Search all cached files."""
    results = {}

    search_term_lower = search_term.lower()

    for filename, entries in TXT_CACHE.items():
        matches = search_txt(entries, search_term_lower)

        if matches:
            results[filename] = matches

    for filename, df in CSV_CACHE.items():
        matches = search_csv(df, search_term)

        if matches:
            results[filename] = matches

    return results


def display_results(search_term, results):
    if not results:
        print(f"\nNo occurrences of '{search_term}' found.\n")
        return

    print(f"\nHere are all the occurrences of '{search_term}':\n")

    for filename in sorted(results):
        print(f"=== {filename} ===\n")

        for french, english in results[filename]:
            print(french)
            print(english)
            print()


def main():
    csv_count, txt_count = load_files()

    print(f"Type '{QUIT_TOKEN}' to quit")
    print(f"Loaded {csv_count} CSV file(s) and " f"{txt_count} TXT file(s).\n")

    while True:
        search_term = input("Enter the word or phrase to search for: ").strip()

        if search_term == QUIT_TOKEN:
            break

        if not search_term:
            continue

        results = search_all(search_term)
        display_results(search_term, results)


if __name__ == "__main__":
    main()
