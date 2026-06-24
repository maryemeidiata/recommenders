"""
matrix_factorization.py
-----------------------
Matrix factorisation via Stochastic Gradient Descent (SGD).

Rather than using a library, I implemented the model from scratch to
understand exactly what's happening under the hood.

The model
---------
Each rating is approximated as:

    r̂(u, i) = μ + b_u + b_i + p_u · q_i

where:
    μ    = global mean rating
    b_u  = user bias (does this user rate higher/lower than average?)
    b_i  = item bias (is this movie rated higher/lower than average?)
    p_u  = user latent factor vector (n_factors dimensions)
    q_i  = item latent factor vector (n_factors dimensions)
    p_u · q_i = dot product capturing user-item affinity

The biases capture systematic tendencies (a harsh critic, a blockbuster film)
while the latent factors capture subtler taste patterns (genre preferences,
preferred tone, etc.).

Training
--------
Parameters are updated with SGD + L2 regularisation to prevent overfitting.
For each observed rating we compute the error e = r - r̂ and take a small
gradient step. The learning rate (lr) and regularisation strength (reg) were
chosen by trial and error on a small validation set.
"""

import numpy as np
import pandas as pd
from . import config
from .data_loading import get_seen_items


class MatrixFactorizationRecommender:
    """
    SGD matrix factorisation with user and item biases.

    Parameters
    ----------
    n_factors : int
        Number of latent dimensions. More factors = more expressive but
        slower to train and more prone to overfitting on small datasets.
        50 worked well on MovieLens-small.
    n_epochs : int
        Number of full passes through the training data.
    lr : float
        SGD learning rate.
    reg : float
        L2 regularisation coefficient applied to all parameters.
    """

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        # Learned parameters (set during fit)
        self.mu_ = None    # global mean
        self.bu_ = None    # user biases, shape (n_users,)
        self.bi_ = None    # item biases, shape (n_items,)
        self.pu_ = None    # user factors, shape (n_users, n_factors)
        self.qi_ = None    # item factors, shape (n_items, n_factors)
        self.user_map_ = None
        self.item_map_ = None
        self.all_items_ = None
        self.train_losses_ = []  # RMSE per epoch, useful for checking convergence

    def fit(self, ratings):
        """Train the model on the provided ratings DataFrame."""
        rng = np.random.default_rng(self.random_state)

        users = ratings[config.USER_COL].unique()
        items = ratings[config.ITEM_COL].unique()

        # Build index mappings for users and items
        self.user_map_ = {u: i for i, u in enumerate(users)}
        self.item_map_ = {it: i for i, it in enumerate(items)}
        self.all_items_ = items

        n_users, n_items = len(users), len(items)

        # Initialise parameters
        self.mu_ = ratings[config.RATING_COL].mean()
        self.bu_ = np.zeros(n_users)
        self.bi_ = np.zeros(n_items)

        # Small random initialisation prevents symmetry in the latent factors
        self.pu_ = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.qi_ = rng.normal(0, 0.1, (n_items, self.n_factors))

        # Convert to numpy array for fast iteration
        rows = ratings[[config.USER_COL, config.ITEM_COL, config.RATING_COL]].values

        self.train_losses_ = []

        for epoch in range(self.n_epochs):
            # Shuffle each epoch so SGD doesn't follow the same sequence repeatedly
            rng.shuffle(rows)

            sq_errors = []
            for u_raw, i_raw, r in rows:
                u = self.user_map_[u_raw]
                i = self.item_map_[i_raw]

                # Predicted rating and error
                pred = self.mu_ + self.bu_[u] + self.bi_[i] + self.pu_[u] @ self.qi_[i]
                err = r - pred
                sq_errors.append(err * err)

                # Gradient descent step for biases
                self.bu_[u] += self.lr * (err - self.reg * self.bu_[u])
                self.bi_[i] += self.lr * (err - self.reg * self.bi_[i])

                # Gradient descent step for latent factors
                # (save pu_[u] before updating, since qi update depends on the old value)
                pu_old = self.pu_[u].copy()
                self.pu_[u] += self.lr * (err * self.qi_[i] - self.reg * self.pu_[u])
                self.qi_[i] += self.lr * (err * pu_old - self.reg * self.qi_[i])

            self.train_losses_.append(float(np.sqrt(np.mean(sq_errors))))

        return self

    def predict_score(self, user_id, item_id):
        """Return the predicted rating for a user-item pair."""
        # Fall back to global mean for unseen users/items
        if user_id not in self.user_map_ or item_id not in self.item_map_:
            return float(self.mu_) if self.mu_ is not None else 0.0

        u = self.user_map_[user_id]
        i = self.item_map_[item_id]
        return float(self.mu_ + self.bu_[u] + self.bi_[i] + self.pu_[u] @ self.qi_[i])

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Score all unseen movies and return the top-n by predicted rating."""
        if user_id not in self.user_map_:
            return []
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        candidates = [item for item in self.all_items_ if item not in seen]
        scores = [(item, self.predict_score(user_id, item)) for item in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scores[:n]]
