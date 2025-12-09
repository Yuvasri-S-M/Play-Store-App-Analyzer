import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import re
from collections import Counter

CLEANED = Path(__file__).resolve().parents[1] / "data" / "cleaned" / "Playstore_cleaned.csv"

def load_df():
    return pd.read_csv(CLEANED, low_memory=False)

def plot_top_categories_installs(df, top_n=15):
    plt.figure(figsize=(12, 6))
    s = df.groupby("Category")["Installs"].sum().sort_values(ascending=False).head(top_n)
    s.plot(kind="bar", color="skyblue")
    plt.title("Top Categories by Total Installs")
    plt.ylabel("Total Installs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_category_avg_rating(df):
    s = df.groupby("Category")["Rating"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    s.plot(kind="bar", color="orange")
    plt.title("Average Rating by Category")
    plt.ylabel("Avg Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_category_app_count(df):
    s = df["Category"].value_counts()

    plt.figure(figsize=(12, 6))
    s.plot(kind="bar", color="green")
    plt.title("App Count by Category")
    plt.ylabel("Number of Apps")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_category_reviews(df):
    s = df.groupby("Category")["Reviews"].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    s.plot(kind="bar", color="purple")
    plt.title("Total Reviews by Category")
    plt.ylabel("Total Reviews")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_free_vs_paid_category(df):
    df = df.copy()
    df["Type"] = df["Price"].apply(lambda x: "Paid" if x > 0 else "Free")

    table = df.groupby(["Category", "Type"]).size().unstack(fill_value=0)

    plt.figure(figsize=(14, 7))
    table.plot(kind="bar", figsize=(14, 7))
    plt.title("Free vs Paid Apps per Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_heatmap_installs_rating(df):
    plt.figure(figsize=(8, 5))
    corr = df[["Installs", "Rating", "Reviews"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap: Installs / Rating / Reviews")
    plt.tight_layout()
    plt.show()

def plot_category_rating_boxplot(df):
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x="Category", y="Rating")
    plt.xticks(rotation=90)
    plt.title("Rating Distribution by Category")
    plt.tight_layout()
    plt.show()

def plot_scatter_installs_vs_reviews(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Installs", y="Reviews", alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Installs vs Reviews (Log Scale)")
    plt.tight_layout()
    plt.show()

def plot_app_name_wordfreq(df, top_n=30):
    text = " ".join(df["App Name"].dropna().astype(str).tolist()).lower()
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text)

    stopwords = {"the","and","for","with","free","pro","app","new","to","of","by","mobile","from"}
    words = [w for w in words if w not in stopwords]

    freq = Counter(words).most_common(top_n)
    labels, counts = zip(*freq)

    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], counts[::-1], color="teal")
    plt.title(f"Top {top_n} Most Common Words in App Names")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_release_trend(df):
    df = df.copy()
    df["Year"] = pd.to_datetime(df["Released"], errors="coerce").dt.year
    s = df["Year"].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    s.plot(kind="line")
    plt.title("App Release Trend by Year")
    plt.ylabel("Number of Apps Released")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.show()

def plot_top_developers(df, top_n=20):
    s = df.groupby("Developer")["Installs"].sum().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    s.plot(kind="bar", color="crimson")
    plt.title("Top Developers by Total Installs")
    plt.ylabel("Total Installs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_all_visuals():
    df = load_df()
    print("\nGenerating ALL visualizations...\n")

    plot_top_categories_installs(df)
    plot_category_avg_rating(df)
    plot_category_app_count(df)
    plot_category_reviews(df)
    plot_free_vs_paid_category(df)

    plot_heatmap_installs_rating(df)
    plot_category_rating_boxplot(df)
    plot_scatter_installs_vs_reviews(df)

    plot_app_name_wordfreq(df)  
    plot_release_trend(df)
    plot_top_developers(df)

if __name__ == "__main__":
    run_all_visuals()


