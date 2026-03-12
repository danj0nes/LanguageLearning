from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time, os
import requests

# ================= CONFIG =================
CSV_FILE = "All French Terms New.csv"
OUTPUT_FILE = "All French Terms New.csv"

FRENCH_COL = "TERM"
EN_COL = "DEFINITION"
TERM_TYPE_COL = "TERM_TYPE"

EXAMPLE_FR_COLS = ["EXAMPLE_1", "EXAMPLE_2", "EXAMPLE_3"]
EXAMPLE_EN_COLS = ["EXAMPLE_EN_1", "EXAMPLE_EN_2", "EXAMPLE_EN_3"]

PROMPT = (
    "Write 3 short French sentences using the word '{}' meaning '{}'. "
    "Return only the sentences separated by newline."
)

MODEL = ""

TRANSLATE_URL = "http://127.0.0.1:5000/translate"

SAVE_INTERVAL = 50
# =========================================

load_dotenv("api.env")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.x.ai/v1",
)

session = requests.Session()

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

        french_text = completion.choices[0].message.content.strip()

        sentences = [s.strip() for s in french_text.split("\n") if s.strip()]

        while len(sentences) < 3:
            sentences.append("")

        sentences = sentences[:3]

        translations = []

        for sentence in sentences:

            r = session.post(
                url=TRANSLATE_URL,
                json={
                    "q": sentence,
                    "source": "fr",
                    "target": "en",
                    "format": "text",
                },
                timeout=10,
            )

            if r.status_code == 200:
                translations.append(r.json()["translatedText"])
            else:
                print("Translation failed:", r.text)
                translations.append("")

        for i in range(len(sentences)):
            df.at[idx, EXAMPLE_FR_COLS[i]] = sentences[i]
            df.at[idx, EXAMPLE_EN_COLS[i]] = translations[i]

        count += 1

        if count % SAVE_INTERVAL == 0:
            df.to_csv(OUTPUT_FILE, index=False)

        time.sleep(0.1)

    except Exception as e:
        print("Error:", e)

df.to_csv(OUTPUT_FILE, index=False)
