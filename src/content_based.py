"""
content_based.py
----------------
Content-based filtering: recommend movies similar to what a user has liked before,
based on movie features (genres and optionally user tags) rather than other users' behaviour.

Approach
--------
1. Represent each movie as a TF-IDF vector of its genre tokens (and optionally tags).
2. Build a user profile as a weighted sum of the vectors of movies they've rated,
   where the weight is (rating - user_mean). This means liked movies pull the profile
   toward their features and disliked movies push it away.
3. Score all unseen movies by cosine similarity to the user profile.

Why TF-IDF instead of raw counts?
   Rare genres (e.g. "Film-Noir") get upweighted relative to common ones (e.g. "Drama"),
   so a user who loves noir films gets more targeted recommendations.

Why centered ratings for the profile?
   Without centering, a user who gives everything 4 stars would have a profile biased
   toward genres of mediocre films they didn't particularly love.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import config
from .data_loading import get_seen_items


class ContentBasedRecommender:
    """
    TF-IDF content-based recommender.

    Parameters
    ----------
    feature_col : str
        Column in the items DataFrame containing feature strings (default: genres).
    use_tags : bool
        If True, append user-generated tags to the genre text before vectorising.
        Tags add personalised vocabulary but also noise; results show a small
        but consistent improvement in precision when tags are included.
    """

    def __init__(self, feature_col=config.GENRES_COL, use_tags=False):
        self.feature_col = feature_col
        self.use_tags = use_tags
        self.vectorizer = TfidfVectorizer()
        self.item_features_ = None   # sparse TF-IDF matrix (n_items × n_features)
        self.item_ids_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings, items, tags=None):
        """
        Build the TF-IDF feature matrix from item metadata.

        MovieLens genres use '|' as a separator (e.g. 'Action|Comedy'),
        so I replace it with a space to make each genre a separate token.
        """
        df = items.copy()
        df["_text"] = df[self.feature_col].str.replace("|", " ", regex=False)

        if self.use_tags and tags is not None:
            # Aggregate all tags written for a movie into a single string
            tag_text = (
                tags.groupby(config.ITEM_COL)["tag"]
                .apply(lambda t: " ".join(t.dropna().astype(str)))
                .reset_index()
                .rename(columns={"tag": "_tags"})
            )
            df = df.merge(tag_text, on=config.ITEM_COL, how="left")
            df["_tags"] = df["_tags"].fillna("")
            df["_text"] = df["_text"] + " " + df["_tags"]

        self.item_ids_ = df[config.ITEM_COL].values
        self.item_id_to_index_ = {item: idx for idx, item in enumerate(self.item_ids_)}
        self.item_features_ = self.vectorizer.fit_transform(df["_text"])
        return self

    def build_user_profile(self, user_id, ratings_train):
        """
        Compute a weighted average feature vector representing this user's taste.

        Formula: profile(u) = Σ_i (r_ui − r̄_u) · v_i
        where r̄_u is the user's mean rating and v_i is the TF-IDF vector of item i.

        The result is L2-normalised so cosine similarity comparisons are consistent
        regardless of how many movies a user has rated.
        """
        user_ratings = ratings_train[ratings_train[config.USER_COL] == user_id]
        if user_ratings.empty:
            return None

        mean_rating = user_ratings[config.RATING_COL].mean()
        profile = np.zeros(self.item_features_.shape[1])

        for _, row in user_ratings.iterrows():
            item_id = row[config.ITEM_COL]
            if item_id not in self.item_id_to_index_:
                continue  # movie exists in ratings but not in items metadata
            idx = self.item_id_to_index_[item_id]
            weight = row[config.RATING_COL] - mean_rating
            profile += weight * self.item_features_[idx].toarray().flatten()

        # Normalise so the profile length doesn't depend on number of ratings
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm
        return profile

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Score all unseen movies by cosine similarity to the user profile."""
        profile = self.build_user_profile(user_id, ratings_train)
        if profile is None:
            return []

        scores = cosine_similarity(profile.reshape(1, -1), self.item_features_).flatten()
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()

        item_scores = [
            (self.item_ids_[i], scores[i])
            for i in range(len(self.item_ids_))
            if self.item_ids_[i] not in seen
        ]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in item_scores[:n]]

    def explain_recommendation(self, user_id, item_id, ratings_train, n=3):
        """
        Return the top-n movies from the user's watch history that most influenced
        this recommendation, based on feature similarity to the recommended item.

        The explanation logic: the recommended item's genre/tag vector is compared
        to every movie the user has rated. Items that are both similar AND highly
        rated (above neutral) surface as the primary explanation.

        Returns
        -------
        list of (movie_id, rating, similarity_score) tuples, sorted by influence.
        """
        if item_id not in self.item_id_to_index_:
            return []

        user_ratings = ratings_train[ratings_train[config.USER_COL] == user_id]
        if user_ratings.empty:
            return []

        item_idx = self.item_id_to_index_[item_id]
        item_vec = self.item_features_[item_idx]

        reasons = []
        for _, row in user_ratings.iterrows():
            rated_id = row[config.ITEM_COL]
            if rated_id not in self.item_id_to_index_ or rated_id == item_id:
                continue
            rated_idx = self.item_id_to_index_[rated_id]
            sim = float(cosine_similarity(item_vec, self.item_features_[rated_idx])[0][0])
            reasons.append((rated_id, float(row[config.RATING_COL]), sim))

        # Rank by similarity x positive rating weight, surfaces liked AND similar movies
        reasons.sort(key=lambda x: x[2] * max(x[1] - 2.5, 0), reverse=True)
        return reasons[:n]

    def similar_items(self, item_id, n=10):
        """Return the n movies most similar to a given movie (item-to-item similarity)."""
        if item_id not in self.item_id_to_index_:
            return []
        idx = self.item_id_to_index_[item_id]
        scores = cosine_similarity(self.item_features_[idx], self.item_features_).flatten()
        ranked = np.argsort(scores)[::-1]
        # Skip rank 0, the movie itself
        return [(self.item_ids_[i], float(scores[i])) for i in ranked[1: n + 1]]
