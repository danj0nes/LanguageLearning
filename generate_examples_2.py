from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time, os

# ================= CONFIG =================
CSV_FILE = "All_Terms.csv"
OUTPUT_FILE = "All_Terms_New.csv"

FRENCH_COL = "TERM"
EN_COL = "DEFINITION"
TERM_TYPE_COL = "TERM_TYPE"

EXAMPLE_FR_COLS = ["EXAMPLE_1", "EXAMPLE_2", "EXAMPLE_3"]
EXAMPLE_EN_COLS = ["EXAMPLE_EN_1", "EXAMPLE_EN_2", "EXAMPLE_EN_3"]

PROMPT = (
    "Write 3 short French sentences to help learn '{}' meaning '{}'. "
    "Use natural and varied grammar around the term."
    "For each sentence provide the English translation."
    "Return exactly 3 lines formatted like:\n"
    "French sentence | English translation"
)

MODEL = "grok-4-fast-non-reasoning"

SAVE_INTERVAL = 10
# =========================================

load_dotenv("api.env")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.x.ai/v1",
)

df = pd.read_csv(CSV_FILE)

# Ensure example columns exist
for col in EXAMPLE_FR_COLS + EXAMPLE_EN_COLS:
    if col not in df.columns:
        df[col] = ""

rows_to_generate = df[
    (df[EXAMPLE_FR_COLS].isna().any(axis=1) | (df[EXAMPLE_FR_COLS] == "").any(axis=1))
    & (df[TERM_TYPE_COL] != "phrase")
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

        lines = [l.strip() for l in response_text.split("\n") if l.strip()]

        fr_sentences = []
        en_sentences = []

        for line in lines:
            if "|" in line:
                fr, en = line.split("|", 1)
                fr_sentences.append(fr.strip())
                en_sentences.append(en.strip())

        while len(fr_sentences) < 3:
            fr_sentences.append("")
            en_sentences.append("")

        fr_sentences = fr_sentences[:3]
        en_sentences = en_sentences[:3]

        for i in range(3):
            df.at[idx, EXAMPLE_FR_COLS[i]] = fr_sentences[i]
            df.at[idx, EXAMPLE_EN_COLS[i]] = en_sentences[i]

        count += 1

        if count % SAVE_INTERVAL == 0:
            df.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print("Error:", e)

df.to_csv(OUTPUT_FILE, index=False)
