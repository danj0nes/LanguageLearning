import pandas as pd

MAIN_TERM_LIST = r"French Vocab.csv"

NEW_TERM_LISTS = [
    r"Lists [138-]/Verbes 27 [140].csv",
    r"Lists [138-]/Noms Courants 37 [141].csv",
    r"Lists [138-]/Noms Courants 38 [142].csv",
    r"Lists [138-]/Verbes 28 [143].csv",
    r"Lists [138-]/Verbes 29 [144].csv",
    r"Lists [138-]/Mots Courants 13 [145].csv",
    r"Lists [138-]/Noms Courants 39 [146].csv",
    r"Lists [138-]/Expressions et Phrases 20 [147].csv",
    r"Lists [138-]/Verbes 30 [148].csv",
    r"Lists [138-]/Verbes 31 [149].csv",
    r"Lists [138-]/Verbes 32 [150].csv",
    r"Lists [138-]/Noms Courants 40 [151].csv",
    r"Lists [138-]/Adjectifs Courant 14 [152].csv",
    r"Lists [138-]/Mots Courants 14 [153].csv",
    r"Lists [138-]/Noms Courants 41 [154].csv",
]


# ============================================================
# Load main term list
# ============================================================

df1 = pd.read_csv(MAIN_TERM_LIST)

if "UNIQUE_ID" not in df1.columns:
    raise ValueError("MAIN_TERM_LIST must contain UNIQUE_ID")

if "LIST_NUMBER" not in df1.columns:
    raise ValueError("MAIN_TERM_LIST must contain LIST_NUMBER")

if "TERM" not in df1.columns:
    raise ValueError("MAIN_TERM_LIST must contain TERM")


# ============================================================
# Determine starting UNIQUE_ID
# ============================================================

next_unique_id = int(df1["UNIQUE_ID"].max()) + 1


# ============================================================
# Existing LIST_NUMBERs
# ============================================================

existing_list_numbers = set(df1["LIST_NUMBER"].dropna())


# ============================================================
# Existing TERM values
# ============================================================

existing_terms = set(df1["TERM"].dropna())


# ============================================================
# Load and validate all new term lists
# ============================================================

new_dataframes = []

used_list_numbers = set(existing_list_numbers)
all_new_terms = set()

for file_path in NEW_TERM_LISTS:

    print(f"Loading: {file_path}")

    df = pd.read_csv(file_path)

    if "TERM" not in df.columns:
        raise ValueError(f"{file_path} must contain TERM")

    # --------------------------------------------------------
    # LIST_NUMBER handling
    # --------------------------------------------------------

    if "LIST_NUMBER" in df.columns:

        unique_values = set(df["LIST_NUMBER"].dropna())

        if len(unique_values) != 1:
            raise ValueError(
                f"{file_path} must contain exactly one " f"LIST_NUMBER value"
            )

        list_number = next(iter(unique_values))

        if list_number in used_list_numbers:
            raise ValueError(
                f"LIST_NUMBER {list_number} from " f"{file_path} already exists"
            )

    else:

        # Generate a new LIST_NUMBER
        numeric_list_numbers = [x for x in used_list_numbers if pd.notna(x)]

        if numeric_list_numbers:
            list_number = max(numeric_list_numbers) + 1
        else:
            list_number = 1

        df["LIST_NUMBER"] = list_number

    # Reserve this LIST_NUMBER so another new file
    # cannot use it.
    used_list_numbers.add(list_number)

    # --------------------------------------------------------
    # Duplicate TERM protection
    # --------------------------------------------------------

    file_terms = set(df["TERM"].dropna())

    duplicates_with_main = file_terms.intersection(existing_terms)

    if duplicates_with_main:
        raise ValueError(
            f"Duplicate TERM values found in {file_path} "
            f"that already exist in MAIN_TERM_LIST: "
            f"{sorted(duplicates_with_main)}"
        )

    duplicates_with_other_new_files = file_terms.intersection(all_new_terms)

    if duplicates_with_other_new_files:
        raise ValueError(
            f"Duplicate TERM values found in {file_path} "
            f"that also occur in another new term list: "
            f"{sorted(duplicates_with_other_new_files)}"
        )

    # Add this file's terms to the global set
    all_new_terms.update(file_terms)

    new_dataframes.append(df)


# ============================================================
# Combine all new term lists
# ============================================================

if not new_dataframes:
    raise ValueError("NEW_TERM_LISTS is empty")

df2 = pd.concat(new_dataframes, ignore_index=True)


# ============================================================
# Assign UNIQUE_IDs to all new rows
# ============================================================

df2["UNIQUE_ID"] = range(next_unique_id, next_unique_id + len(df2))


# ============================================================
# Align columns with main term list
# ============================================================

for column in df1.columns:

    if column not in df2.columns:
        df2[column] = pd.NA


# Keep exactly the same column order as the main list
df2 = df2[df1.columns]


# ============================================================
# Append
# ============================================================

combined = pd.concat([df1, df2], ignore_index=True)


# ============================================================
# Enforce types
# ============================================================

for col in ["UNIQUE_ID", "LIST_NUMBER"]:

    if col in combined.columns:

        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")


for col in ["DEF_DATE_LAST_TESTED", "TERM_DATE_LAST_TESTED"]:

    if col in combined.columns:

        combined[col] = pd.to_datetime(
            combined[col], errors="coerce", format="mixed"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ============================================================
# Save
# ============================================================

combined.to_csv(MAIN_TERM_LIST, index=False)


# ============================================================
# Report
# ============================================================

print()
print("Successfully added:")
print(f"  Files: {len(NEW_TERM_LISTS)}")
print(f"  Rows:  {len(df2)}")
print(f"  UNIQUE_IDs: " f"{next_unique_id}-" f"{next_unique_id + len(df2) - 1}")

print()
print("LIST_NUMBERs added:")

for df in new_dataframes:
    list_number = df["LIST_NUMBER"].iloc[0]
    print(f"  {list_number}: {len(df)} rows")
