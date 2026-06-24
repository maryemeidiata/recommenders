"""
collaborative_filtering.py
--------------------------
Memory-based collaborative filtering: item-item and user-user variants.

The core idea is that similar users like similar things, and similar items
are liked by the same users. We don't need any content metadata at all;
just the pattern of ratings in the user-item matrix.

Both models use cosine similarity because it works well on sparse,
positive-valued rating data and is fast to compute with scikit-learn.

Item-item vs. User-user
-----------------------
Item-item CF tends to be more stable over time because items don't change
their behaviour the way users do. It also scales better when there are
many more users than items (as is typical). User-user CF can be more
personalised but suffers more from sparsity.

I implemented both so we can directly compare them on the same dataset.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from . import config
from .data_loading import get_seen_items


def _build_user_item_matrix(ratings):
    """
    Convert the ratings DataFrame into a mean-centered sparse user-item matrix.

    Mean-centering subtracts each user's average rating from their individual
    ratings before computing similarity. This removes per-user rating scale bias:
    a user who gives everything 4 stars looks different from one who gives 2 to 3 stars,
    even if they have identical taste. Without centering, cosine similarity conflates
    magnitude (generous rater) with direction (actual preference).

    Returns
    -------
    matrix     : csr_matrix  (n_users × n_items), values are mean-centered ratings
    user_ids   : array of original user IDs
    item_ids   : array of original item IDs
    user_map   : dict mapping user_id → row index
    item_map   : dict mapping item_id → col index
    user_means : dict mapping user_id → mean rating (used to de-center predictions)
    """
    user_ids = ratings[config.USER_COL].unique()
    item_ids = ratings[config.ITEM_COL].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {it: i for i, it in enumerate(item_ids)}

    user_means = ratings.groupby(config.USER_COL)[config.RATING_COL].mean().to_dict()

    rows = ratings[config.USER_COL].map(user_map)
    cols = ratings[config.ITEM_COL].map(item_map)
    # Centered rating: r_ui - r̄_u
    data = ratings[config.RATING_COL].values - ratings[config.USER_COL].map(user_means).values

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return matrix, user_ids, item_ids, user_map, item_map, user_means


class ItemItemCollaborativeFiltering:
    """
    Item-item collaborative filtering.

    For each unrated movie, we predict a score by looking at the k most
    similar movies that the user HAS rated, and taking a similarity-weighted
    average of those ratings.

    score(u, i) = Σ_j sim(i, j) · r(u, j)  /  Σ_j |sim(i, j)|
    summed over the top-k similar items j already rated by user u.
    """

    def __init__(self, k=20):
        # k = number of similar items to consider when predicting a score
        self.k = k
        self.user_item_matrix_ = None
        self.item_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_map_ = None
        self.item_map_ = None
        self.user_means_ = None

    def fit(self, ratings):
        """Build the mean-centered user-item matrix and pre-compute all item-item similarities."""
        matrix, user_ids, item_ids, user_map, item_map, user_means = _build_user_item_matrix(ratings)
        self.user_item_matrix_ = matrix
        self.user_ids_ = user_ids
        self.item_ids_ = item_ids
        self.user_map_ = user_map
        self.item_map_ = item_map
        self.user_means_ = user_means

        # Items are columns, so transpose: each row of matrix.T is one item's centered rating vector
        self.item_similarity_ = cosine_similarity(matrix.T)
        return self

    def predict_score(self, user_id, item_id):
        """
        Predict how much user_id would like item_id.

        The matrix contains mean-centered ratings (r_uj - r̄_u), so the
        weighted average gives a centered prediction. We add back r̄_u
        to get a rating on the original scale.
        """
        if user_id not in self.user_map_ or item_id not in self.item_map_:
            return 0.0

        u_idx = self.user_map_[user_id]
        i_idx = self.item_map_[item_id]

        sims = self.item_similarity_[i_idx]
        user_row = self.user_item_matrix_[u_idx].toarray().flatten()

        # Non-zero entries are rated items (a centered zero means rated exactly at mean,
        # which contributes nothing to the prediction and can safely be excluded)
        rated_mask = user_row != 0
        rated_mask[i_idx] = False  # exclude the target item itself

        rated_sims = sims[rated_mask]
        centered_ratings = user_row[rated_mask]

        if len(rated_sims) == 0:
            return self.user_means_.get(user_id, 0.0)

        # Keep only the top-k most similar rated items
        top_k = min(self.k, len(rated_sims))
        top_idx = np.argpartition(rated_sims, -top_k)[-top_k:]
        w, r_centered = rated_sims[top_idx], centered_ratings[top_idx]
        denom = np.abs(w).sum()
        if denom == 0:
            return self.user_means_.get(user_id, 0.0)
        # De-center: add back user mean to get a rating on the original scale
        return float(self.user_means_.get(user_id, 0.0) + np.dot(w, r_centered) / denom)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Score all unseen movies and return the top-n."""
        if user_id not in self.user_map_:
            return []
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        candidates = [item for item in self.item_ids_ if item not in seen]
        scores = [(item, self.predict_score(user_id, item)) for item in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scores[:n]]


class UserUserCollaborativeFiltering:
    """
    User-user collaborative filtering.

    Instead of comparing items, we find the k users most similar to the
    target user and aggregate their ratings for movies the target hasn't seen.

    score(u, i) = Σ_v sim(u, v) · r(v, i)  /  Σ_v |sim(u, v)|
    summed over the top-k similar users v who have rated item i.

    This can be more personalised than item-item but is more sensitive to
    sparsity: if similar users have not rated a movie, we cannot score it.
    """

    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix_ = None
        self.user_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_map_ = None
        self.item_map_ = None
        self.user_means_ = None

    def fit(self, ratings):
        """Build the mean-centered user-item matrix and pre-compute all user-user similarities."""
        matrix, user_ids, item_ids, user_map, item_map, user_means = _build_user_item_matrix(ratings)
        self.user_item_matrix_ = matrix
        self.user_ids_ = user_ids
        self.item_ids_ = item_ids
        self.user_map_ = user_map
        self.item_map_ = item_map
        self.user_means_ = user_means

        # Each row is a mean-centered user vector; cosine similarity between rows = user-user similarity
        self.user_similarity_ = cosine_similarity(matrix)
        return self

    def predict_score(self, user_id, item_id):
        """
        Predict how much user_id would like item_id based on similar users.

        The matrix contains mean-centered ratings (r_vi - r̄_v) for each neighbour v.
        The weighted average of these centered values gives an unbiased estimate of
        how much above or below average the target user would rate this item.
        We then add the target user's own mean to restore the scale.
        """
        if user_id not in self.user_map_ or item_id not in self.item_map_:
            return 0.0

        u_idx = self.user_map_[user_id]
        i_idx = self.item_map_[item_id]

        sims = self.user_similarity_[u_idx]

        # Only consider users who have actually rated item_id (non-zero centered value)
        item_col = self.user_item_matrix_[:, i_idx].toarray().flatten()
        rated_mask = item_col != 0
        rated_mask[u_idx] = False  # don't include the target user in their own prediction

        rated_sims = sims[rated_mask]
        centered_ratings = item_col[rated_mask]

        if len(rated_sims) == 0:
            return self.user_means_.get(user_id, 0.0)

        top_k = min(self.k, len(rated_sims))
        top_idx = np.argpartition(rated_sims, -top_k)[-top_k:]
        w, r_centered = rated_sims[top_idx], centered_ratings[top_idx]
        denom = np.abs(w).sum()
        if denom == 0:
            return self.user_means_.get(user_id, 0.0)
        return float(self.user_means_.get(user_id, 0.0) + np.dot(w, r_centered) / denom)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        if user_id not in self.user_map_:
            return []
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        candidates = [item for item in self.item_ids_ if item not in seen]
        scores = [(item, self.predict_score(user_id, item)) for item in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scores[:n]]
