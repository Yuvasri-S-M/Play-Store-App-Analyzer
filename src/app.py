import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import io
import re
from collections import Counter
from pathlib import Path
import time
import joblib

import numpy as _np
def safe_log1p(X):
    X = _np.array(X, dtype=float)
    mask = ~_np.isfinite(X)
    if mask.any():
        X[mask] = 0.0
    return _np.log1p(X)

st.set_page_config(page_title="Play Store App Analyzer", layout="wide")

BASE = Path(__file__).resolve().parents[1]
CLEANED = BASE / "data" / "cleaned" / "Playstore_cleaned.csv"
SAMPLE = BASE / "data" / "cleaned" / "Playstore_sample.csv"
MODEL_PATH = BASE / "models" / "app_success_clf.pkl"

st.markdown(
    """
    <style>
    /* Reduce top padding on main header area */
    .css-18e3th9 { padding-top: 0.6rem; padding-bottom: 0.35rem; }
    /* Reduce spacing inside expanders */
    .streamlit-expanderHeader { padding-top: .25rem; padding-bottom: .25rem; }
    /* Reduce default block padding */
    .css-1d391kg { padding-top: .2rem; padding-bottom: .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_sample(path: Path = SAMPLE) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def load_full_with_progress(path: Path = CLEANED, chunksize: int = 100000):
    reader = pd.read_csv(path, low_memory=False, chunksize=chunksize)
    parts = []
    total_rows = 0
    try:
        size_bytes = path.stat().st_size
        approx_chunks = max(1, int(size_bytes / (chunksize * 200)))
    except Exception:
        approx_chunks = None

    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, part in enumerate(reader, start=1):
        parts.append(part)
        total_rows += len(part)
        if approx_chunks:
            pct = int(min(100, (i / approx_chunks) * 100))
        else:
            pct = min(100, int((i * 5) % 100))
        progress_bar.progress(pct)
        status_text.info(f"Rows read so far: {total_rows:,}")
        time.sleep(0.03)

    df = pd.concat(parts, ignore_index=True)
    progress_bar.progress(100)
    status_text.success(f"Finished reading {len(df):,} rows.")
    return df

def top_wordfreq(df, col='App Name', top_n=30):
    text = " ".join(df[col].dropna().astype(str).tolist()).lower()
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
    stopwords = {"the","and","for","with","free","pro","app","new","to","of","by","mobile","from"}
    words = [w for w in words if w not in stopwords]
    freq = Counter(words).most_common(top_n)
    if freq:
        labels, counts = zip(*freq)
    else:
        labels, counts = [], []
    return labels, counts

def download_button_excel(df):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button("Download filtered data (XLSX)", data=towrite, file_name="filtered_playstore.xlsx")

@st.cache_resource
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        return {"_load_error": str(e)}

_model_cached = load_model()
def get_model():
    return _model_cached

st.title("📱 Play Store App Analyzer")

if "full_df" in st.session_state and isinstance(st.session_state["full_df"], pd.DataFrame):
    df = st.session_state["full_df"]
    using_full = True
    st.caption(f"Full dataset loaded ({len(df):,} rows) — loaded in this session.")
else:
    df = load_sample()
    using_full = False
    st.caption(f"Using sample dataset ({len(df):,} rows) for fast UI. Click 'Load full dataset' to load all rows.")

st.sidebar.header("Controls")
if using_full:
    st.sidebar.write("Full dataset already loaded in session.")
load_btn = st.sidebar.button("Load full dataset (large, may take time)")

if load_btn:
    st.sidebar.info("Loading full dataset; please wait and do not refresh the page.")
    full_df = load_full_with_progress()
    st.session_state["full_df"] = full_df
    df = full_df

for col in ["Installs", "Rating", "Reviews", "Price"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

st.sidebar.subheader("Filters")
category_list = sorted(df['Category'].dropna().unique()) if 'Category' in df.columns else []
category = st.sidebar.multiselect("Category", options=category_list, default=[])
dev_options = []
if 'Developer' in df.columns:
    top_devs = df.groupby('Developer')['Installs'].sum().sort_values(ascending=False).head(200).index.tolist()
    dev_options = top_devs
devs = st.sidebar.multiselect("Developer (top 200)", options=dev_options, default=[])
price_type = st.sidebar.selectbox("Type", options=["All", "Free", "Paid"], index=0)
min_installs = st.sidebar.number_input("Min installs", min_value=0, value=1000, step=1000)
min_rating = st.sidebar.slider("Min rating", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
top_n = st.sidebar.slider("Top N results", 5, 50, 10)

filtered = df.copy()
if category:
    filtered = filtered[filtered['Category'].isin(category)]
if devs:
    filtered = filtered[filtered['Developer'].isin(devs)]
if price_type != "All" and 'Price' in filtered.columns:
    if price_type == "Free":
        filtered = filtered[(filtered['Price'].isna()) | (filtered['Price'] == 0)]
    else:
        filtered = filtered[filtered['Price'] > 0]
if 'Installs' in filtered.columns:
    filtered = filtered[filtered['Installs'] >= min_installs]
if 'Rating' in filtered.columns:
    filtered = filtered[filtered['Rating'] >= min_rating]

k1, k2, k3, k4 = st.columns(4, gap="small")
k1.metric("Rows (filtered)", f"{len(filtered):,}")
k2.metric("Unique Categories", f"{filtered['Category'].nunique():,}" if 'Category' in filtered.columns else "N/A")
k3.metric("Total Installs (filtered)", f"{int(filtered['Installs'].sum()):,}" if 'Installs' in filtered.columns else "N/A")
k4.metric("Avg Rating", f"{filtered['Rating'].mean():.2f}" if 'Rating' in filtered.columns else "N/A")

left_col, ml_col = st.columns([2, 1], gap="medium")

with left_col:

    with st.expander("Top N apps by installs", expanded=True):
        cols_to_show = [c for c in ['App Name','Developer','Category','Installs','Rating','Reviews','Price'] if c in filtered.columns]
        st.dataframe(
            filtered.sort_values("Installs", ascending=False).head(top_n)[cols_to_show],
            use_container_width=True
        )

    st.subheader("Top Categories by Installs")
    if 'Category' in filtered.columns and 'Installs' in filtered.columns:
        cat_installs = (
            filtered.groupby('Category')['Installs']
            .sum().reset_index()
            .sort_values('Installs', ascending=False)
            .head(15)
        )
        st.altair_chart(
            alt.Chart(cat_installs).mark_bar().encode(
                x=alt.X('Installs:Q', title='Total Installs'),
                y=alt.Y('Category:N', sort='-x', title=None),
                tooltip=["Category", alt.Tooltip("Installs", format=",")]
            ).properties(height=340),
            use_container_width=True
        )
    else:
        st.write("Category or Installs column missing.")

    st.subheader("Rating Distribution")
    if 'Rating' in filtered.columns:
        hist = alt.Chart(filtered.dropna(subset=['Rating'])).mark_bar().encode(
            alt.X("Rating:Q", bin=alt.Bin(maxbins=20)),
            y='count()'
        ).properties(height=160)
        st.altair_chart(hist, use_container_width=True)
    else:
        st.write("No rating data.")

    st.subheader("Installs vs Reviews (log-log)")
    if 'Installs' in filtered.columns and 'Reviews' in filtered.columns:
        scatter = alt.Chart(filtered.dropna(subset=['Installs','Reviews'])).transform_calculate(
            log_installs="log(datum.Installs + 1)",
            log_reviews="log(datum.Reviews + 1)"
        ).mark_circle(opacity=0.35).encode(
            x='log_installs:Q',
            y='log_reviews:Q',
            color='Category:N' if 'Category' in filtered.columns else alt.value('steelblue'),
            tooltip=['App Name','Developer','Installs','Reviews','Rating'] if 'App Name' in filtered.columns else ['Installs','Reviews']
        ).properties(height=380)
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.write("Need Installs and Reviews for scatter.")

    st.subheader("Top words in app names")
    labels, counts = top_wordfreq(filtered, top_n=25)
    if labels:
        plt.figure(figsize=(8,4.5))
        plt.barh(labels[::-1], counts[::-1])
        plt.tight_layout()
        st.pyplot(plt.gcf())
    else:
        st.write("No words to show.")

    st.subheader("Apps released by year")
    if 'Released' in filtered.columns:
        year_counts = filtered['Released'].pipe(lambda s: pd.to_datetime(s, errors='coerce')).dt.year.value_counts().sort_index()
        if not year_counts.empty:
            st.line_chart(year_counts)
        else:
            st.write("No release-year data available.")
    else:
        st.write("Released column missing.")

    st.subheader("Top Developers by Installs")
    if 'Developer' in filtered.columns and 'Installs' in filtered.columns:
        top_devs = filtered.groupby('Developer')['Installs'].sum().sort_values(ascending=False).head(20)
        st.bar_chart(top_devs)
    else:
        st.write("Developer or Installs column missing.")

with ml_col:
    st.markdown("### 📈 ML App Success Predictor")

    model = get_model()
    if model is None:
        st.info("No trained model found. Train it with: `python src/train_model.py --sample`")

    elif isinstance(model, dict) and "_load_error" in model:
        st.error("Model load error:")
        st.write(model["_load_error"])

    else:
        pred_category = st.selectbox("Category", category_list if category_list else ["Unknown"])
        pred_rating = st.slider("Rating", 0.0, 5.0, 4.0, 0.1)
        pred_reviews = st.number_input("Reviews", min_value=0, value=1000, step=100)
        pred_price = st.number_input("Price (USD)", min_value=0.0, value=0.0, step=0.01)
        pred_dev_top = st.selectbox("Is Developer Top Ranked?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        if st.button("Predict with ML"):
            input_df = pd.DataFrame([{
                "Category": pred_category,
                "Rating": float(pred_rating),
                "Reviews": float(pred_reviews),
                "Price": float(pred_price),
                "DeveloperTop": int(pred_dev_top)
            }])
            try:
                prob = float(model.predict_proba(input_df)[:, 1][0])
                st.metric("Probability of 1M+ Installs", f"{prob:.4f}")
                if prob > 0.75:
                    st.success("🔥 High chance of success!")
                elif prob > 0.45:
                    st.warning("⚠️ Medium chance — needs improvement.")
                else:
                    st.error("❌ Low chance of reaching 1M installs.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with st.expander("Export / Download"):
    st.write("Download the currently filtered table.")
    download_button_excel(filtered)


st.write("---")
st.caption("Built with Streamlit. Layout B: KPIs on top, left = main (2/3), right = ML predictor (1/3).")


