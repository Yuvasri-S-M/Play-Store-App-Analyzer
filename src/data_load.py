import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_FILE = DATA_DIR / "Playstore_final.csv"

def load_raw(path: str | Path = DEFAULT_FILE) -> pd.DataFrame:
    """Load CSV using a tolerant parser (skips malformed rows)."""
    path = Path(path)
    print(f"Loading {path} ...")
    df = pd.read_csv(
        path,
        engine="python",        
        on_bad_lines="skip",    
        encoding="utf-8"
    )
    print("Loaded shape:", df.shape)
    return df

if __name__ == "__main__":
    df = load_raw()
    print(df.head())


