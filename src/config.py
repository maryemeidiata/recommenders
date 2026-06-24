"""
config.py
---------
Central place for all paths and constants used across the project.
Keeping everything here means I only have to change a path once
if I switch datasets or move files around.
"""

from pathlib import Path

# ── Project layout ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Dataset file paths (MovieLens Latest Small) ───────────────────────────────
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"
ITEMS_PATH = RAW_DATA_DIR / "movies.csv"
TAGS_PATH = RAW_DATA_DIR / "tags.csv"

# ── Column name constants ──────────────────────────────────────────────────────
# Centralising column names avoids hard-coded strings scattered across files.
USER_COL = "userId"
ITEM_COL = "movieId"
RATING_COL = "rating"
TIMESTAMP_COL = "timestamp"
TITLE_COL = "title"
GENRES_COL = "genres"

# ── Recommendation settings ────────────────────────────────────────────────────
TOP_K = 10        # default list length shown to users
RANDOM_STATE = 42  # for reproducibility across all random operations
