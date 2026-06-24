"""
data_loading.py
---------------
Handles everything related to loading and preparing the MovieLens dataset.

I chose a temporal train/test split over a random one because it better
reflects a real deployment scenario: we always predict future ratings
from past behaviour, never the other way around. A random split would
leak future information into training, giving overly optimistic results.
"""

import pandas as pd
from . import config


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ratings(path=config.RATINGS_PATH):
    """
    Load the ratings CSV and do a quick sanity-check on column names.
    MovieLens format: userId, movieId, rating (0.5 to 5.0 half-stars), timestamp.
    """
    df = pd.read_csv(path)

    required = {config.USER_COL, config.ITEM_COL, config.RATING_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ratings file is missing expected columns: {missing}")

    return df


def load_items(path=config.ITEMS_PATH):
    """
    Load the movies CSV.
    Each movie has a title (with year) and a pipe-separated genres string,
    e.g. 'Action|Adventure|Sci-Fi'.
    """
    df = pd.read_csv(path)

    required = {config.ITEM_COL, config.TITLE_COL, config.GENRES_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Items file is missing expected columns: {missing}")

    return df


# ── Exploratory Data Analysis ──────────────────────────────────────────────────

def describe_dataset(ratings, items=None):
    """
    Print a structured summary of the dataset.

    Sparsity tells us how empty the user-item matrix is.
    At 98% sparsity the matrix is almost entirely zeros,
    which is exactly why collaborative filtering is hard:
    most user-item pairs have never been observed.
    """
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_ratings = len(ratings)

    # Sparsity = fraction of user-item pairs with NO observed rating
    sparsity = 1 - n_ratings / (n_users * n_items)

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Users:        {n_users:,}")
    print(f"Items:        {n_items:,}")
    print(f"Ratings:      {n_ratings:,}")
    print(f"Sparsity:     {sparsity:.4%}")
    print()

    print("Rating distribution:")
    print(ratings[config.RATING_COL].value_counts().sort_index().to_string())
    print()

    # Most frequently rated items (popular != highly rated)
    most_rated = (
        ratings.groupby(config.ITEM_COL)[config.RATING_COL]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={config.RATING_COL: "n_ratings"})
    )
    if items is not None:
        most_rated = most_rated.merge(
            items[[config.ITEM_COL, config.TITLE_COL]], on=config.ITEM_COL, how="left"
        )
    print("Top 10 most rated items:")
    print(most_rated.to_string(index=False))
    print()

    # Most active users, these users disproportionately shape the model
    most_active = (
        ratings.groupby(config.USER_COL)[config.RATING_COL]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={config.RATING_COL: "n_ratings"})
    )
    print("Top 10 most active users:")
    print(most_active.to_string(index=False))
    print("=" * 50)

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
    }


# ── Train / test split ────────────────────────────────────────────────────────

def train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE):
    """
    Temporal split: for each user, the most recent (test_size * 100)% of
    their ratings go to the test set and the rest to training.

    This mirrors how a real system would be evaluated: you train on history
    and predict what the user will rate next, not a random subset of past ratings.
    """
    # Sort by user then time so each user's ratings are chronological
    ratings = ratings.sort_values([config.USER_COL, config.TIMESTAMP_COL])

    def split_user(group):
        n = len(group)
        # At least one rating always stays in training
        cutoff = max(1, int(n * (1 - test_size)))
        group = group.reset_index(drop=True)
        group["_split"] = "train"
        group.loc[cutoff:, "_split"] = "test"
        return group

    rated = ratings.groupby(config.USER_COL, group_keys=False).apply(
        split_user, include_groups=False
    )

    # Re-attach the user column that was excluded by include_groups=False
    rated[config.USER_COL] = ratings.sort_values(
        [config.USER_COL, config.TIMESTAMP_COL]
    )[config.USER_COL].values

    train = rated[rated["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    test = rated[rated["_split"] == "test"].drop(columns="_split").reset_index(drop=True)

    # Drop test rows for users that somehow have no training data
    test = test[test[config.USER_COL].isin(train[config.USER_COL])]

    return train, test


def get_seen_items(ratings, user_id):
    """Return the set of movie IDs that a user has already rated."""
    return set(ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL])
