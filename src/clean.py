import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = BASE_DIR / "data" / "raw" / "Playstore_final.csv"
OUT_DIR = BASE_DIR / "data" / "cleaned"

if OUT_DIR.exists() and not OUT_DIR.is_dir():
    print(f"{OUT_DIR} exists as a FILE, removing it...")
    os.remove(OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_playstore_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    if "Installs" in df.columns:
        df["Installs"] = (
            df["Installs"].astype(str)
            .str.replace("+", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    
        df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

    if "Price" in df.columns:
        df["Price"] = df["Price"].astype(str).str.replace("$", "", regex=False).str.strip()
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    if "Reviews" in df.columns:
        df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    df = df.dropna(subset=["Installs", "Rating"])

    df = df.reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Loading raw file:", RAW_FILE)

    df = pd.read_csv(
        RAW_FILE,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8"
    )
    cleaned = clean_playstore_df(df)

    out_file = OUT_DIR / "Playstore_cleaned.csv"
    cleaned.to_csv(out_file, index=False)

    print("Cleaned files saved to:", out_file)


