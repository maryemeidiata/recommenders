"""
baselines.py
------------
Non-personalised recommenders that serve as performance benchmarks.

These models ignore who the user is; they just recommend globally
popular or highly rated movies. They're surprisingly hard to beat,
especially for new users with little history (the cold-start problem).

Implemented:
    - MostPopularRecommender    : rank by interaction count
    - HighestAverageRatingRecommender : rank by mean rating (with minimum threshold)
    - RandomRecommender         : random baseline (lower-bound sanity check)
"""

import numpy as np
import pandas as pd
from . import config
from .data_loading import get_seen_items


class MostPopularRecommender:
    """
    Recommend the movies that have been rated most often across all users.

    Popularity is measured by interaction count, not average rating.
    A film with 10,000 ratings at 3.5 stars beats one with 50 ratings at 5.0.
    This reflects what people actually watch, not just what critics love.
    """

    def __init__(self):
        self.ranking_ = None  # ordered list of item IDs, most popular first

    def fit(self, ratings, items=None):
        # Count how many ratings each movie received, then sort descending
        self.ranking_ = (
            ratings.groupby(config.ITEM_COL)[config.RATING_COL]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Return the top-n most popular movies the user hasn't seen yet."""
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        recs = [item for item in self.ranking_ if item not in seen]
        return recs[:n]


class HighestAverageRatingRecommender:
    """
    Recommend movies with the highest average rating, filtered by a minimum
    number of ratings to avoid obscure films with a handful of perfect scores.

    The min_ratings threshold is a simple but effective quality gate.
    Without it, a movie with one 5-star rating would top every list.
    """

    def __init__(self, min_ratings=20):
        # Minimum number of ratings a movie must have to be considered
        self.min_ratings = min_ratings
        self.ranking_ = None

    def fit(self, ratings, items=None):
        stats = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(["mean", "count"])

        # Only keep movies with enough data to trust the average
        qualified = stats[stats["count"] >= self.min_ratings]
        self.ranking_ = qualified["mean"].sort_values(ascending=False).index.tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        recs = [item for item in self.ranking_ if item not in seen]
        return recs[:n]


class RandomRecommender:
    """
    Recommend random unseen movies.

    Useful as a lower-bound baseline; any sensible algorithm should
    comfortably outperform random recommendations.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        self.items_ = ratings[config.ITEM_COL].unique().tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        rng = np.random.default_rng(self.random_state)
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        candidates = [i for i in self.items_ if i not in seen]
        n = min(n, len(candidates))
        return rng.choice(candidates, size=n, replace=False).tolist()
