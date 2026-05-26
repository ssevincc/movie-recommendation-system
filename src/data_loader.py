from pathlib import Path
from zipfile import ZipFile

import pandas as pd


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def load_movielens_1m(data_dir="data"):
    """Load MovieLens 1M files from data/ml-1m."""
    dataset_dir = Path(data_dir) / "ml-1m"

    ratings = pd.read_csv(
        dataset_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        dataset_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    users = pd.read_csv(
        dataset_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    movies["clean_title"] = movies["title"].str.replace(r"\s+\(\d{4}\)$", "", regex=True)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype("float")

    return ratings, movies, users


def extract_zip(zip_path, data_dir="data"):
    """Extract the MovieLens zip file into the data directory."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

