from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time, os

# ================= CONFIG =================
CSV_FILE = "All_Terms_2.csv"
OUTPUT_FILE = "All_Terms_New_3.csv"

FRENCH_COL = "TERM"
EN_COL = "DEFINITION"
TERM_TYPE_COL = "TERM_TYPE"

IPA_COL = "IPA"

PROMPT = (
    "Write the IPA pronunciation for the French word/phrase '{}' meaning '{}'. "
    "Use standard IPA notation. "
    "Return only the IPA pronunciation, enclosed in slashes."
)

MODEL = "grok-4-fast-reasoning"

SAVE_INTERVAL = 10
# =========================================

load_dotenv("api.env")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.x.ai/v1",
)

df = pd.read_csv(CSV_FILE)

# Ensure example columns exist
if IPA_COL not in df.columns:
    df[IPA_COL] = ""

rows_to_generate = df[
    (df[IPA_COL].isna() | (df[IPA_COL] == "")) & (df[TERM_TYPE_COL] != "phrase")
]


def retry(func, attempts=3):
    for i in range(attempts):
        try:
            return func()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(1)


count = 0

for idx, row in tqdm(rows_to_generate.iterrows(), total=rows_to_generate.shape[0]):

    french_term = row[FRENCH_COL]
    english_def = row[EN_COL]

    try:
        completion = retry(
            lambda: client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": PROMPT.format(french_term, english_def)}
                ],
                max_tokens=100,
            )
        )

        response_text = completion.choices[0].message.content.strip()

        df.at[idx, IPA_COL] = response_text.strip()

        count += 1

        if count % SAVE_INTERVAL == 0:
            df.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print("Error:", e)

df.to_csv(OUTPUT_FILE, index=False)
