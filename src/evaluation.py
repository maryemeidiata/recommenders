"""
evaluation.py
-------------
Evaluation metrics for ranking-based recommender systems.

All metrics operate at a cut-off K, meaning we only look at the top-K
items returned by the model, reflecting real usage where users only
scroll through a limited list.

Accuracy metrics
----------------
- Precision@K : what fraction of the top-K recommendations are relevant?
- Recall@K    : what fraction of all relevant items appear in the top-K?
- NDCG@K      : like precision, but rewards hitting relevant items earlier in the list.
- MRR         : where does the first relevant item appear? (useful when one good rec is enough)
- Hit Rate@K  : binary, returns 1 if at least one relevant item appears in the top-K

Beyond-accuracy metrics
-----------------------
- Catalog Coverage : what share of all available movies are recommended to at least one user?
  Low coverage = popularity bias (the model recommends the same blockbusters to everyone).
- Novelty@K        : how surprising are the recommendations? Measured via self-information
  of popularity; less popular items score higher.

These extra metrics matter because a model that only recommends Forrest Gump
to everyone would have decent accuracy but terrible coverage and novelty.
"""

import numpy as np
import pandas as pd
from . import config


def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Fraction of the top-k recommendations that are actually relevant.
    Penalises recommending irrelevant items regardless of where relevant ones land.
    """
    recs = list(recommended_items)[:k]
    if not recs:
        return 0.0
    return sum(1 for item in recs if item in set(relevant_items)) / k


def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Fraction of all relevant items that appear in the top-k.
    Important when the user wants to discover everything relevant, not just one thing.
    """
    recs = list(recommended_items)[:k]
    rel = set(relevant_items)
    if not rel:
        return 0.0
    return sum(1 for item in recs if item in rel) / len(rel)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    """
    Returns 1 if at least one relevant item appears in the top-k, else 0.
    A coarser signal than precision but easy to interpret.
    """
    return 1.0 if set(list(recommended_items)[:k]) & set(relevant_items) else 0.0


def dcg_at_k(relevance_scores, k=10):
    """
    Discounted Cumulative Gain: sum of relevance scores discounted by rank.
    Items appearing earlier in the list contribute more to the score.

    DCG = Σ rel_i / log2(i + 1),  i starting at 1.
    """
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(list(relevance_scores)[:k]))


def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    Normalised DCG: DCG divided by the ideal DCG (best possible ordering).
    Returns a value in [0, 1], where 1 means perfect ranking of relevant items.
    """
    recs = list(recommended_items)[:k]
    rel_set = set(relevant_items)
    relevance = [1 if item in rel_set else 0 for item in recs]

    dcg = dcg_at_k(relevance, k)
    # Ideal DCG: all relevant items appear at the top
    ideal = dcg_at_k([1] * min(len(rel_set), k), k)
    return dcg / ideal if ideal > 0 else 0.0


def mean_reciprocal_rank(recommended_items, relevant_items, k=10):
    """
    Reciprocal rank of the first relevant item.
    Useful for tasks where one good recommendation is sufficient (e.g. search).
    Returns 0 if no relevant item appears in the top-k.
    """
    rel_set = set(relevant_items)
    for rank, item in enumerate(list(recommended_items)[:k], start=1):
        if item in rel_set:
            return 1.0 / rank
    return 0.0


def catalog_coverage(all_recommendations, all_items):
    """
    Fraction of the full movie catalog that appears in at least one recommendation list.

    A model with low coverage suffers from popularity bias; it keeps recommending
    the same mainstream titles and ignores the long tail of niche films.
    """
    recommended = set()
    for recs in all_recommendations:
        recommended.update(recs)
    return len(recommended) / len(set(all_items)) if all_items else 0.0


def novelty_at_k(recommended_items, item_popularity, n_users, k=10):
    """
    Mean self-information of recommended items.

    Self-information of item i = -log2(popularity(i) / n_users).
    Rare items have high self-information (= high novelty).
    Popular items everyone has seen have low novelty.

    This captures whether the model pushes beyond blockbusters.
    """
    recs = list(recommended_items)[:k]
    scores = []
    for item in recs:
        pop = item_popularity.get(item, 1)
        p = pop / n_users
        scores.append(-np.log2(p) if p > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


RELEVANCE_THRESHOLD = 3.5  # ratings below this are not treated as positive feedback


def intra_list_diversity(recommended_items, item_features, item_id_to_index, k=10):
    """
    Mean pairwise dissimilarity of the top-k recommended items.

    Computed as 1 − average cosine similarity over all pairs in the list.
    Score of 1.0 means completely dissimilar items (maximum diversity);
    0.0 means every item is identical in feature space.

    This measures whether an algorithm tends to recommend a narrow genre slice
    (low diversity) or a varied selection of movies (high diversity).
    Requires a feature matrix, typically the content-based TF-IDF vectors.
    """
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    indices = [item_id_to_index[item] for item in list(recommended_items)[:k]
               if item in item_id_to_index]
    if len(indices) < 2:
        return 0.0
    vecs = item_features[indices]
    sims = _cos_sim(vecs)
    n = len(indices)
    upper = sims[np.triu_indices(n, k=1)]
    return float(1.0 - np.mean(upper))


def evaluate_model(model, ratings_train, ratings_test, users, k=10):
    """
    Run evaluation for a model across a sample of users and return averaged metrics.

    Only test-set ratings >= RELEVANCE_THRESHOLD are treated as ground-truth relevant
    items. This avoids counting low-rated movies as "things the user liked," which
    would distort every accuracy metric.
    Metrics are averaged over all users that have at least one qualifying test rating.
    """
    metrics = {"precision": [], "recall": [], "ndcg": [], "mrr": [], "hit_rate": []}

    for user_id in users:
        user_test = ratings_test[ratings_test[config.USER_COL] == user_id]
        relevant = set(
            user_test.loc[user_test[config.RATING_COL] >= RELEVANCE_THRESHOLD, config.ITEM_COL]
        )
        if not relevant:
            continue  # skip users with no positively-rated test items

        recs = model.recommend(user_id, ratings_train, n=k)
        metrics["precision"].append(precision_at_k(recs, relevant, k))
        metrics["recall"].append(recall_at_k(recs, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recs, relevant, k))
        metrics["mrr"].append(mean_reciprocal_rank(recs, relevant, k))
        metrics["hit_rate"].append(hit_rate_at_k(recs, relevant, k))

    return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in metrics.items()}
