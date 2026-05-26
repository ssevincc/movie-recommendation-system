from pathlib import Path
import sys
import urllib.request
import zipfile

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data_loader import load_movielens_1m
from src.recommender import HybridMovieRecommender

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "ml-1m"
ZIP_PATH = DATA_DIR / "ml-1m.zip"

st.set_page_config(page_title="Movie Recommender", layout="wide")


def download_movielens():
    DATA_DIR.mkdir(exist_ok=True)

    if DATASET_DIR.exists():
        return

    with st.spinner("Downloading MovieLens 1M dataset..."):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    with st.spinner("Extracting dataset..."):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)


@st.cache_data
def load_data():
    download_movielens()
    return load_movielens_1m(DATA_DIR)


@st.cache_resource
def train_model():
    ratings, movies, users = load_data()
    model = HybridMovieRecommender(n_components=50, random_state=42)
    model.fit(ratings, movies)
    return model, ratings, movies, users


st.title("Hybrid Movie Recommendation System")
st.caption("MovieLens 1M | Collaborative filtering + content-based filtering")

model, ratings, movies, users = train_model()

mode = st.sidebar.radio(
    "Recommendation mode",
    ["Hybrid user recommendations", "Collaborative user recommendations", "Similar movies"],
)

top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if mode == "Similar movies":
    movie_title = st.selectbox("Choose a movie", movies["title"].sort_values().tolist())

    if st.button("Recommend similar movies", type="primary"):
        recommendations = model.recommend_similar_movies(movie_title, top_n=top_n)
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

elif mode == "Collaborative user recommendations":
    user_id = st.number_input(
        "User ID",
        min_value=int(ratings["user_id"].min()),
        max_value=int(ratings["user_id"].max()),
        value=1,
        step=1,
    )

    if st.button("Recommend for user", type="primary"):
        recommendations = model.recommend_for_user(user_id=user_id, top_n=top_n)
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

else:
    user_id = st.number_input(
        "User ID",
        min_value=int(ratings["user_id"].min()),
        max_value=int(ratings["user_id"].max()),
        value=1,
        step=1,
    )

    collaborative_weight = st.slider(
        "Collaborative filtering weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
    )

    if st.button("Recommend for user", type="primary"):
        recommendations = model.hybrid_recommend_for_user(
            user_id=user_id,
            top_n=top_n,
            collaborative_weight=collaborative_weight,
        )
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
