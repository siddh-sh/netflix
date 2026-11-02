import json
import sys
import time
from pathlib import Path

import joblib
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import monitoring  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.train import METRICS_PATH, MODEL_PATH, train  # noqa: E402

st.set_page_config(page_title="Netflix Analytics", page_icon="🎬", layout="wide")


@st.cache_resource
def init_metrics():
    monitoring.start_metrics_server()
    return True


@st.cache_data
def get_data():
    return load_clean()


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


init_metrics()
df = get_data()
model, model_metrics = get_model()
monitoring.DATASET_ROWS.set(len(df))
monitoring.MODEL_ACCURACY.set(model_metrics["accuracy"])
monitoring.MODEL_F1.set(model_metrics["f1_macro"])

st.title("🎬 Netflix Titles — Analysis & ML")
page = st.sidebar.radio("Page", ["Overview", "Trends", "Explore", "Predict"])
monitoring.PAGE_VIEWS.labels(page=page).inc()
st.sidebar.caption(f"Model v{model_metrics['version']} · acc {model_metrics['accuracy']:.2%}")

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titles", f"{len(df):,}")
    c2.metric("Movies", f"{(df['type'] == 'Movie').sum():,}")
    c3.metric("TV Shows", f"{(df['type'] == 'TV Show').sum():,}")
    c4.metric("Countries", df["primary_country"].nunique())

    left, right = st.columns(2)
    left.plotly_chart(px.pie(df, names="type", title="Movies vs TV Shows", hole=0.4), use_container_width=True)
    ratings = df["rating"].value_counts().reset_index()
    right.plotly_chart(px.bar(ratings, x="rating", y="count", title="Ratings"), use_container_width=True)

    countries = df[df["primary_country"] != "Unknown"]["primary_country"].value_counts().head(10).reset_index()
    st.plotly_chart(px.bar(countries, x="count", y="primary_country", orientation="h",
                           title="Top 10 countries").update_yaxes(autorange="reversed"), use_container_width=True)

elif page == "Trends":
    added = df.dropna(subset=["year_added"]).groupby(["year_added", "type"]).size().reset_index(name="count")
    st.plotly_chart(px.line(added, x="year_added", y="count", color="type", markers=True,
                            title="Titles added per year"), use_container_width=True)

    release = df.groupby(["release_year", "type"]).size().reset_index(name="count")
    st.plotly_chart(px.area(release[release["release_year"] >= 1980], x="release_year", y="count", color="type",
                            title="Titles by release year"), use_container_width=True)

    genres = df["listed_in"].str.split(", ").explode().value_counts().head(15).reset_index()
    st.plotly_chart(px.bar(genres, x="count", y="listed_in", orientation="h", title="Top 15 genres")
                    .update_yaxes(autorange="reversed"), use_container_width=True)

    movies = df[df["type"] == "Movie"]
    st.plotly_chart(px.histogram(movies, x="duration_value", nbins=50, title="Movie duration (minutes)"),
                    use_container_width=True)

elif page == "Explore":
    col1, col2, col3 = st.columns(3)
    types = col1.multiselect("Type", df["type"].unique().tolist(), default=df["type"].unique().tolist())
    countries = col2.multiselect("Country", sorted(df["primary_country"].unique()))
    years = col3.slider("Release year", int(df["release_year"].min()), int(df["release_year"].max()), (2010, 2021))
    query = st.text_input("Search title")

    view = df[df["type"].isin(types) & df["release_year"].between(*years)]
    if countries:
        view = view[view["primary_country"].isin(countries)]
    if query:
        view = view[view["title"].str.contains(query, case=False, na=False)]
    st.caption(f"{len(view):,} titles")
    st.dataframe(view[["title", "type", "primary_country", "release_year", "rating", "duration", "listed_in"]],
                 use_container_width=True, hide_index=True)

else:
    st.subheader("Predict Movie vs TV Show")
    st.write("A TF-IDF + Logistic Regression model trained on title descriptions and genres.")
    description = st.text_area("Description", "A group of friends navigate love and life across several seasons.")
    genres = st.text_input("Genres", "TV Comedies, Romantic TV Shows")
    if st.button("Predict"):
        start = time.perf_counter()
        proba = model.predict_proba([f"{description} {genres}"])[0]
        monitoring.PREDICTION_LATENCY.observe(time.perf_counter() - start)
        idx = proba.argmax()
        label = model.classes_[idx]
        monitoring.PREDICTIONS.labels(label=label).inc()
        monitoring.PREDICTION_CONFIDENCE.observe(proba[idx])
        st.success(f"**{label}** ({proba[idx]:.1%} confidence)")
    st.json(model_metrics)
