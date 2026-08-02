import pandas as pd

MAIN_TERM_LIST = r"20260802_export.csv"
NEW_TERM_LIST = r"Lists [138-]/Verbes 26 [139].csv"

# Load files
df1 = pd.read_csv(MAIN_TERM_LIST)
df2 = pd.read_csv(NEW_TERM_LIST)

# NEED TO INFORCE DATA TYPES!!!


# ----- UNIQUE_ID handling -----

if "UNIQUE_ID" not in df1.columns:
    raise ValueError("MAIN_TERM_LIST must contain UNIQUE_ID")

next_unique_id = int(df1["UNIQUE_ID"].max()) + 1

df2["UNIQUE_ID"] = range(next_unique_id, next_unique_id + len(df2))

# ----- LIST_NUMBER handling -----

if "LIST_NUMBER" not in df1.columns:
    raise ValueError("CSV_FILE_1 must contain LIST_NUMBER")

existing_list_numbers = set(df1["LIST_NUMBER"].dropna())

if "LIST_NUMBER" in df2.columns:

    unique_values = set(df2["LIST_NUMBER"].dropna())

    if len(unique_values) != 1:
        raise ValueError("CSV_FILE_2 must contain exactly one LIST_NUMBER value")

    list_number = next(iter(unique_values))

    if list_number in existing_list_numbers:
        raise ValueError(f"LIST_NUMBER {list_number} already exists in CSV_FILE_1")

else:
    list_number = int(df1["LIST_NUMBER"].max()) + 1
    df2["LIST_NUMBER"] = list_number

# ----- Duplicate protection -----

duplicate_terms = set(df1["TERM"]).intersection(set(df2["TERM"]))

if duplicate_terms:
    raise ValueError(f"Duplicate TERM values found: {sorted(duplicate_terms)}")

# ----- Align columns -----

for column in df1.columns:
    if column not in df2.columns:
        df2[column] = pd.NA

df2 = df2[df1.columns]

# ----- Append and save -----

combined = pd.concat([df1, df2], ignore_index=True)

combined.to_csv(MAIN_TERM_LIST, index=False)

print(
    f"Added {len(df2)} rows "
    f"with LIST_NUMBER={list_number} "
    f"and UNIQUE_IDs {next_unique_id}-{next_unique_id + len(df2) - 1}"
)
