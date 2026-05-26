# %% [markdown]
# # Hybrid Movie Recommendation System
#
# CV-ready baseline using MovieLens 1M:
# - Content-based recommendation using movie genres
# - Collaborative filtering using matrix factorization
# - Hybrid recommendations combining both approaches
# - Evaluation with RMSE, Precision@K, Recall@K, and NDCG@K

# %%
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

RANDOM_STATE = 42
DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "ml-1m"


# %% [markdown]
# ## 1. Download and Extract Dataset
# In Google Colab, run this cell. The dataset is downloaded from the official GroupLens source.

# %%
if not DATASET_DIR.exists():
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "ml-1m.zip"

    if not zip_path.exists():
        !wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip -O data/ml-1m.zip

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

print("Dataset ready:", DATASET_DIR)


# %% [markdown]
# ## 2. Load Data

# %%
ratings = pd.read_csv(
    DATASET_DIR / "ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
    encoding="latin-1",
)

movies = pd.read_csv(
    DATASET_DIR / "movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1",
)

users = pd.read_csv(
    DATASET_DIR / "users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip_code"],
    encoding="latin-1",
)

ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
movies["clean_title"] = movies["title"].str.replace(r"\s+\(\d{4}\)$", "", regex=True)
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype("float")

print("Ratings:", ratings.shape)
print("Movies:", movies.shape)
print("Users:", users.shape)
movies.head()


# %% [markdown]
# ## 3. Basic EDA

# %%
print(ratings["rating"].describe())
print("\nMost rated movies:")
ratings.merge(movies, on="movie_id").groupby("title")["rating"].agg(["count", "mean"]).sort_values(
    "count", ascending=False
).head(10)


# %% [markdown]
# ## 4. Content-Based Recommender
# This recommends movies similar to a selected title using genres.

# %%
movies["genre_text"] = movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies["genre_text"])
content_similarity = cosine_similarity(genre_matrix)

movie_index = pd.Series(movies.index, index=movies["title"]).drop_duplicates()


def recommend_similar_movies(title, top_n=10):
    if title not in movie_index:
        matches = movies[movies["title"].str.contains(title, case=False, na=False)]["title"].head(10)
        raise ValueError(f"Movie not found. Possible matches: {list(matches)}")

    idx = movie_index[title]
    similarity_scores = list(enumerate(content_similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in similarity_scores[1 : top_n + 1]]

    return movies.loc[top_indices, ["movie_id", "title", "genres"]].reset_index(drop=True)


recommend_similar_movies("Toy Story (1995)", top_n=10)


# %% [markdown]
# ## 5. Collaborative Filtering with Matrix Factorization
# We create a user-movie rating matrix and use TruncatedSVD to learn latent factors.

# %%
train_ratings, test_ratings = train_test_split(
    ratings,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=ratings["rating"],
)

global_mean = train_ratings["rating"].mean()

user_movie_matrix = train_ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
)

user_means = user_movie_matrix.mean(axis=1)
centered_matrix = user_movie_matrix.sub(user_means, axis=0).fillna(0)

n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
user_factors = svd.fit_transform(centered_matrix)
movie_factors = svd.components_.T

predicted_centered = np.dot(user_factors, movie_factors.T)
predicted_ratings = predicted_centered + user_means.values.reshape(-1, 1)
predicted_ratings = np.clip(predicted_ratings, 1, 5)

predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns,
)

print("Prediction matrix:", predicted_df.shape)


def predict_rating(user_id, movie_id):
    if user_id in predicted_df.index and movie_id in predicted_df.columns:
        return float(predicted_df.loc[user_id, movie_id])

    if movie_id in train_ratings["movie_id"].values:
        return float(train_ratings.loc[train_ratings["movie_id"] == movie_id, "rating"].mean())

    return float(global_mean)


test_sample = test_ratings.copy()
test_sample["prediction"] = test_sample.apply(
    lambda row: predict_rating(row["user_id"], row["movie_id"]),
    axis=1,
)

mse = mean_squared_error(test_sample["rating"], test_sample["prediction"])
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")


# %% [markdown]
# ## 6. Collaborative Recommendations

# %%
def recommend_for_user(user_id, top_n=10):
    user_id = int(user_id)

    if user_id not in predicted_df.index:
        raise ValueError("Unknown user_id")

    watched_movie_ids = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"])
    user_predictions = predicted_df.loc[user_id].sort_values(ascending=False)
    user_predictions = user_predictions.drop(labels=list(watched_movie_ids), errors="ignore")

    recommendations = (
        user_predictions.head(top_n)
        .rename("predicted_rating")
        .reset_index()
        .merge(movies, on="movie_id")
    )

    return recommendations[["movie_id", "title", "genres", "predicted_rating"]]


recommend_for_user(user_id=1, top_n=10)


# %% [markdown]
# ## 7. Hybrid Recommender
# Combines:
# - collaborative predicted score
# - content similarity to movies the user liked

# %%
movie_id_to_content_idx = dict(zip(movies["movie_id"], movies.index))


def get_user_content_scores(user_id):
    user_id = int(user_id)

    user_history = ratings[ratings["user_id"] == user_id]
    liked_movies = user_history[user_history["rating"] >= 4]["movie_id"]
    liked_indices = [
        movie_id_to_content_idx[movie_id]
        for movie_id in liked_movies
        if movie_id in movie_id_to_content_idx
    ]

    if not liked_indices:
        return pd.Series(0, index=movies["movie_id"])

    mean_similarity = content_similarity[liked_indices].mean(axis=0)
    return pd.Series(mean_similarity, index=movies["movie_id"])


def min_max_scale(series):
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def hybrid_recommend_for_user(user_id, top_n=10, collaborative_weight=0.7):
    user_id = int(user_id)

    if user_id not in predicted_df.index:
        raise ValueError("Unknown user_id")

    watched_movie_ids = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"])

    collaborative_scores = predicted_df.loc[user_id]
    content_scores = get_user_content_scores(user_id)

    collaborative_scores = min_max_scale(collaborative_scores)
    content_scores = min_max_scale(content_scores)

    common_movie_ids = collaborative_scores.index.intersection(content_scores.index)

    hybrid_scores = (
        collaborative_weight * collaborative_scores.loc[common_movie_ids]
        + (1 - collaborative_weight) * content_scores.loc[common_movie_ids]
    )

    hybrid_scores = hybrid_scores.drop(labels=list(watched_movie_ids), errors="ignore")

    recommendations = (
        hybrid_scores.sort_values(ascending=False)
        .head(top_n)
        .rename("hybrid_score")
        .reset_index()
        .merge(movies, on="movie_id")
    )

    return recommendations[["movie_id", "title", "genres", "hybrid_score"]]


hybrid_recommend_for_user(user_id=1, top_n=10)


# %% [markdown]
# ## 8. Ranking Evaluation
# For recommendation systems, Precision@K, Recall@K, and NDCG@K are more useful than accuracy.
# We treat ratings >= 4 as relevant movies.

# %%
def precision_at_k(recommended_ids, relevant_ids, k):
    recommended_k = recommended_ids[:k]
    if len(recommended_k) == 0:
        return 0.0
    return len(set(recommended_k) & set(relevant_ids)) / k


def recall_at_k(recommended_ids, relevant_ids, k):
    if len(relevant_ids) == 0:
        return np.nan
    recommended_k = recommended_ids[:k]
    return len(set(recommended_k) & set(relevant_ids)) / len(relevant_ids)


def ndcg_at_k(recommended_ids, relevant_ids, k):
    recommended_k = recommended_ids[:k]
    dcg = 0.0
    for i, movie_id in enumerate(recommended_k):
        if movie_id in relevant_ids:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return np.nan

    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def evaluate_recommender(user_ids, k=10):
    rows = []

    for user_id in user_ids:
        relevant_ids = set(
            test_ratings.loc[
                (test_ratings["user_id"] == user_id) & (test_ratings["rating"] >= 4),
                "movie_id",
            ]
        )

        if len(relevant_ids) == 0 or user_id not in predicted_df.index:
            continue

        recommended_ids = hybrid_recommend_for_user(user_id, top_n=k)["movie_id"].tolist()

        rows.append(
            {
                "user_id": user_id,
                f"precision@{k}": precision_at_k(recommended_ids, relevant_ids, k),
                f"recall@{k}": recall_at_k(recommended_ids, relevant_ids, k),
                f"ndcg@{k}": ndcg_at_k(recommended_ids, relevant_ids, k),
            }
        )

    return pd.DataFrame(rows)


evaluation_users = test_ratings["user_id"].drop_duplicates().sample(300, random_state=RANDOM_STATE)
ranking_metrics = evaluate_recommender(evaluation_users, k=10)
ranking_metrics.mean(numeric_only=True)


# %% [markdown]
# ## 9. Search Helper for Demo

# %%
def search_movies(query, limit=10):
    result = movies[movies["title"].str.contains(query, case=False, na=False)]
    return result[["movie_id", "title", "genres"]].head(limit)


search_movies("matrix")


# %% [markdown]
# ## 10. Example Outputs for README

# %%
print("Content-based recommendations for Toy Story:")
display(recommend_similar_movies("Toy Story (1995)", top_n=10))

print("Hybrid recommendations for user 1:")
display(hybrid_recommend_for_user(user_id=1, top_n=10))

print("RMSE:", round(rmse, 4))
display(ranking_metrics.mean(numeric_only=True).to_frame("score"))
