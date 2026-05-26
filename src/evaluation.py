import numpy as np
from sklearn.metrics import mean_squared_error


def rmse_score(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return np.sqrt(mse)


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

