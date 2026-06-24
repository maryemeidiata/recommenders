"""
main.py
-------
Orchestrates the full recommender pipeline:
    1. Load MovieLens data
    2. EDA summary
    3. Temporal train/test split
    4. Fit all models
    5. Evaluate with Precision, Recall, NDCG, MRR, Hit Rate, Coverage, Novelty
    6. Print recommendation examples for 3 users
    7. Save results to results/metrics.csv

Run with:  python3 main.py
"""

import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_ratings, load_items, train_test_split_ratings, describe_dataset
from src.baselines import MostPopularRecommender, HighestAverageRatingRecommender, RandomRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model, catalog_coverage, novelty_at_k, intra_list_diversity

# ── Settings ──────────────────────────────────────────────────────────────────
K = 10           # recommendation list length
EVAL_USERS = 200  # evaluate on a sample to keep runtime manageable


def show_recs(recs, items, label):
    """Print a numbered recommendation list with movie titles."""
    title_map = items.set_index(config.ITEM_COL)[config.TITLE_COL]
    print(f"\n  [{label}]")
    for i, item_id in enumerate(recs, 1):
        print(f"  {i:2d}. {title_map.get(item_id, item_id)}")


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    ratings = load_ratings()
    items = load_items()
    tags = pd.read_csv(config.TAGS_PATH)

    # ── 2. Exploratory data analysis ──────────────────────────────────────────
    stats = describe_dataset(ratings, items)

    # ── 3. Temporal train/test split ──────────────────────────────────────────
    # Using a temporal split: the most recent 20% of each user's ratings
    # go to the test set. This avoids leaking future data into training.
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    print(f"\nTrain: {len(train):,} ratings | Test: {len(test):,} ratings")

    # Random sample of test users for representative evaluation
    test_users = test[config.USER_COL].unique()
    rng = np.random.default_rng(config.RANDOM_STATE)
    eval_users = rng.choice(test_users, size=min(EVAL_USERS, len(test_users)), replace=False)

    # Precompute item popularity for novelty metric
    item_popularity = train.groupby(config.ITEM_COL)[config.RATING_COL].count().to_dict()
    n_users = train[config.USER_COL].nunique()
    all_items = items[config.ITEM_COL].tolist()

    # ── 4. Fit all models ─────────────────────────────────────────────────────
    print("\nFitting models...")
    models = {
        "random":             RandomRecommender(),
        "most_popular":       MostPopularRecommender(),
        "highest_avg":        HighestAverageRatingRecommender(min_ratings=20),
        "content_based":      ContentBasedRecommender(use_tags=False),
        "content_based_tags": ContentBasedRecommender(use_tags=True),
        "item_item_cf":       ItemItemCollaborativeFiltering(k=20),
        "user_user_cf":       UserUserCollaborativeFiltering(k=20),
        "matrix_fact":        MatrixFactorizationRecommender(n_factors=50, n_epochs=20),
    }

    for name, model in models.items():
        print(f"  Training {name}...")
        if name.startswith("content_based"):
            model.fit(train, items, tags)
        elif name in ("item_item_cf", "user_user_cf", "matrix_fact"):
            model.fit(train)
        else:
            model.fit(train, items)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print(f"\nEvaluating on {len(eval_users)} users at K={K}...")
    results = {}

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        m = evaluate_model(model, train, test, eval_users, k=K)

        # Beyond-accuracy metrics
        recs_list = [model.recommend(u, train, n=K) for u in eval_users]
        m["coverage"] = catalog_coverage(recs_list, all_items)
        m["novelty"] = float(np.mean([
            novelty_at_k(r, item_popularity, n_users, K) for r in recs_list
        ]))
        cb_model = models["content_based"]
        m["diversity"] = float(np.mean([
            intra_list_diversity(r, cb_model.item_features_, cb_model.item_id_to_index_, K)
            for r in recs_list
        ]))
        results[name] = m

    # ── 6. Results table ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T.round(4)
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(results_df.to_string())
    results_df.to_csv(config.RESULTS_DIR / "metrics.csv")
    print(f"\nSaved to {config.RESULTS_DIR / 'metrics.csv'}")

    # ── 7. Recommendation examples for 3 users ────────────────────────────────
    sample_users = list(eval_users[:3])
    print("\n" + "=" * 70)
    print("RECOMMENDATION EXAMPLES")
    print("=" * 70)

    for user_id in sample_users:
        n_train = len(train[train[config.USER_COL] == user_id])
        print(f"\nUser {user_id}  ({n_train} ratings in training set)")
        for name, model in models.items():
            recs = model.recommend(user_id, train, n=5)
            show_recs(recs, items, name)


if __name__ == "__main__":
    main()
