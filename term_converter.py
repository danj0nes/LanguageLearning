import pandas as pd
from pathlib import Path

FILE_NAME = r"20260918_export.csv"

ALLOWED_COLUMNS = [
    "DEFINITION",
    "DEF_DATE_LAST_TESTED",
    "DEF_LEARNT_SCORE",
    "EXAMPLE_1",
    "EXAMPLE_1_DEF",
    "EXAMPLE_2",
    "EXAMPLE_2_DEF",
    "EXAMPLE_3",
    "EXAMPLE_3_DEF",
    "IPA",
    "LIST_NUMBER",
    "TERM",
    "TERM_DATE_LAST_TESTED",
    "TERM_LEARNT_SCORE",
    "TERM_TYPE",
    "UNIQUE_ID",
]

script_dir = Path(__file__).parent
input_path = script_dir / FILE_NAME
output_path = input_path.with_name(f"{input_path.stem}_converted.csv")

df = pd.read_csv(input_path)

# Keep only columns that exist
df = df[[col for col in ALLOWED_COLUMNS if col in df.columns]]

# Add average learnt score columns
df["TERM_AVG_LEARNT_SCORE"] = df.get("TERM_LEARNT_SCORE")
df["DEF_AVG_LEARNT_SCORE"] = df.get("DEF_LEARNT_SCORE")

# Enforce types
for col in ["UNIQUE_ID", "LIST_NUMBER"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
