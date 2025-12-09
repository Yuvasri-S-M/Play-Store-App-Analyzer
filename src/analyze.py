import pandas as pd
from pathlib import Path

CLEANED = Path(__file__).resolve().parents[1] / "data" / "cleaned" / "Playstore_cleaned.csv"

def load_cleaned(path: Path = CLEANED) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df

def detect_app_column(df: pd.DataFrame) -> str | None:
    cols = df.columns.tolist()
    if 'App' in cols:
        return 'App'
    for c in cols:
        if 'app' in c.lower() or 'name' in c.lower():
            return c
    return None

def top_n_by_installs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    app_col = detect_app_column(df)
    cols = []
    if app_col:
        cols.append(app_col)
    if 'Category' in df.columns:
        cols.append('Category')
    if 'Installs' in df.columns:
        cols.append('Installs')
    if 'Rating' in df.columns:
        cols.append('Rating')
    if 'Reviews' in df.columns:
        cols.append('Reviews')
    if 'Installs' not in df.columns:
        raise ValueError("Installs column missing; cannot compute top apps.")
    return df.sort_values("Installs", ascending=False).head(n)[cols]

def top_categories_by_installs(df: pd.DataFrame, n: int = 10) -> pd.Series:
    if 'Category' in df.columns and 'Installs' in df.columns:
        return df.groupby('Category', dropna=False)['Installs'].sum().sort_values(ascending=False).head(n)
    return pd.Series(dtype='float64')

def free_vs_paid_summary(df: pd.DataFrame) -> pd.DataFrame:
    if 'Price' not in df.columns:
        print("No 'Price' column found — cannot compute Free vs Paid summary.")
        return pd.DataFrame()
    df = df.copy()
    df['Type'] = df['Price'].apply(lambda x: 'Paid' if pd.notna(x) and float(x) > 0 else 'Free')
    agg = df.groupby('Type').agg(
        Count=('Type','count'),
        TotalInstalls=('Installs','sum') if 'Installs' in df.columns else pd.NamedAgg(column='Installs', aggfunc='sum'),
        AvgRating=('Rating','mean') if 'Rating' in df.columns else pd.NamedAgg(column='Rating', aggfunc='mean')
    )
    return agg

if __name__ == "__main__":
    df = load_cleaned()
    print("Loaded cleaned CSV shape:", df.shape)
    print("\nColumns found:\n", df.columns.tolist())

    # Top 10 apps
    try:
        print("\nTop 10 apps by installs:")
        print(top_n_by_installs(df, 10).to_string(index=False))
    except Exception as e:
        print("Could not compute top apps:", e)

    # Top categories
    print("\nTop categories by total installs:")
    cats = top_categories_by_installs(df, 15)
    if not cats.empty:
        print(cats.to_string())
    else:
        print("Category/Installs columns missing or empty.")

    # Free vs Paid summary
    print("\nFree vs Paid summary:")
    fv = free_vs_paid_summary(df)
    if not fv.empty:
        print(fv.to_string())
    else:
        print("No summary produced (maybe missing Price column).")


