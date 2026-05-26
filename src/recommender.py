import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class HybridMovieRecommender:
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, ratings, movies):
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        self.global_mean = self.ratings["rating"].mean()

        self._fit_content_model()
        self._fit_collaborative_model()
        return self

    def _fit_content_model(self):
        self.movies["genre_text"] = self.movies["genres"].str.replace("|", " ", regex=False)

        self.tfidf = TfidfVectorizer()
        self.genre_matrix = self.tfidf.fit_transform(self.movies["genre_text"])
        self.content_similarity = cosine_similarity(self.genre_matrix)

        self.movie_index = pd.Series(self.movies.index, index=self.movies["title"]).drop_duplicates()
        self.movie_id_to_content_idx = dict(zip(self.movies["movie_id"], self.movies.index))

    def _fit_collaborative_model(self):
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings,
            test_size=0.2,
            random_state=self.random_state,
            stratify=self.ratings["rating"],
        )

        self.user_movie_matrix = self.train_ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
        )

        self.user_means = self.user_movie_matrix.mean(axis=1)
        centered_matrix = self.user_movie_matrix.sub(self.user_means, axis=0).fillna(0)

        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        user_factors = self.svd.fit_transform(centered_matrix)
        movie_factors = self.svd.components_.T

        predicted_centered = np.dot(user_factors, movie_factors.T)
        predicted_ratings = predicted_centered + self.user_means.values.reshape(-1, 1)
        predicted_ratings = np.clip(predicted_ratings, 1, 5)

        self.predicted_df = pd.DataFrame(
            predicted_ratings,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.columns,
        )

    def search_movies(self, query, limit=10):
        result = self.movies[self.movies["title"].str.contains(query, case=False, na=False)]
        return result[["movie_id", "title", "genres"]].head(limit)

    def recommend_similar_movies(self, title, top_n=10):
        if title not in self.movie_index:
            matches = self.search_movies(title, limit=10)["title"].tolist()
            raise ValueError(f"Movie not found. Possible matches: {matches}")

        idx = self.movie_index[title]
        similarity_scores = list(enumerate(self.content_similarity[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in similarity_scores[1 : top_n + 1]]

        return self.movies.loc[top_indices, ["movie_id", "title", "genres"]].reset_index(drop=True)

    def predict_rating(self, user_id, movie_id):
        user_id = int(user_id)
        movie_id = int(movie_id)

        if user_id in self.predicted_df.index and movie_id in self.predicted_df.columns:
            return float(self.predicted_df.loc[user_id, movie_id])

        if movie_id in self.train_ratings["movie_id"].values:
            return float(self.train_ratings.loc[self.train_ratings["movie_id"] == movie_id, "rating"].mean())

        return float(self.global_mean)

    def recommend_for_user(self, user_id, top_n=10):
        user_id = int(user_id)

        if user_id not in self.predicted_df.index:
            raise ValueError("Unknown user_id")

        watched_movie_ids = set(self.ratings.loc[self.ratings["user_id"] == user_id, "movie_id"])
        user_predictions = self.predicted_df.loc[user_id].sort_values(ascending=False)
        user_predictions = user_predictions.drop(labels=list(watched_movie_ids), errors="ignore")

        recommendations = (
            user_predictions.head(top_n)
            .rename("predicted_rating")
            .reset_index()
            .merge(self.movies, on="movie_id")
        )

        return recommendations[["movie_id", "title", "genres", "predicted_rating"]]

    def get_user_content_scores(self, user_id):
        user_id = int(user_id)

        user_history = self.ratings[self.ratings["user_id"] == user_id]
        liked_movies = user_history[user_history["rating"] >= 4]["movie_id"]
        liked_indices = [
            self.movie_id_to_content_idx[movie_id]
            for movie_id in liked_movies
            if movie_id in self.movie_id_to_content_idx
        ]

        if not liked_indices:
            return pd.Series(0, index=self.movies["movie_id"])

        mean_similarity = self.content_similarity[liked_indices].mean(axis=0)
        return pd.Series(mean_similarity, index=self.movies["movie_id"])

    @staticmethod
    def min_max_scale(series):
        min_value = series.min()
        max_value = series.max()
        if max_value == min_value:
            return pd.Series(0, index=series.index)
        return (series - min_value) / (max_value - min_value)

    def hybrid_recommend_for_user(self, user_id, top_n=10, collaborative_weight=0.7):
        user_id = int(user_id)

        if user_id not in self.predicted_df.index:
            raise ValueError("Unknown user_id")

        watched_movie_ids = set(self.ratings.loc[self.ratings["user_id"] == user_id, "movie_id"])

        collaborative_scores = self.min_max_scale(self.predicted_df.loc[user_id])
        content_scores = self.min_max_scale(self.get_user_content_scores(user_id))
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
            .merge(self.movies, on="movie_id")
        )

        return recommendations[["movie_id", "title", "genres", "hybrid_score"]]
