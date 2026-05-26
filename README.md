# Hybrid Movie Recommendation System

This project builds a movie recommendation system using the MovieLens 1M dataset. It combines collaborative filtering and content-based filtering to generate personalized movie recommendations.

## Project Features

- Content-based recommendations using movie genres
- Collaborative filtering using matrix factorization with SVD
- Hybrid recommendations combining both approaches
- Evaluation with RMSE, Precision@K, Recall@K, and NDCG@K
- Streamlit app for interactive recommendations

## Dataset

Dataset: MovieLens 1M  
Source: https://grouplens.org/datasets/movielens/1m/

The dataset is not included in this repository. Download `ml-1m.zip`, extract it, and place the folder like this:

```text
data/
  ml-1m/
    movies.dat
    ratings.dat
    users.dat
```

## Project Structure

```text
movie-recommender-system/
  app/
    streamlit_app.py
  src/
    data_loader.py
    evaluation.py
    recommender.py
  movie_recommender_colab.py
  README.md
  requirements.txt
  .gitignore
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Methods

The project uses three recommendation approaches:

- Content-based filtering recommends movies with similar genres.
- Collaborative filtering predicts user ratings from user-movie interaction patterns.
- Hybrid filtering combines collaborative and content-based scores.

## Evaluation Metrics

- RMSE for rating prediction quality
- Precision@K for top-K recommendation relevance
- Recall@K for coverage of relevant movies
- NDCG@K for ranking quality
