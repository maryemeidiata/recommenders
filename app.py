"""
app.py
------
Streamlit UI for the MovieLens recommender prototype.
Auto-downloads MovieLens Latest Small on first run.
Run locally:  streamlit run app.py
"""

import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cinemetric: Movie Recommender",
    page_icon=":film_frames:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide sidebar toggle & default header clutter ── */
[data-testid="collapsedControl"] { display: none; }
header { background: transparent !important; }

/* ── Top header bar ── */
.top-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1rem 0 1.2rem;
    border-bottom: 1px solid #1e1e3a;
    margin-bottom: 1.5rem;
}
.top-bar .logo-icon { font-size: 1.6rem; color: #e94560; }
.top-bar .logo-name { font-size: 1.4rem; font-weight: 700; color: white; letter-spacing: -0.5px; }
.top-bar .logo-sub  { font-size: 0.72rem; color: rgba(255,255,255,0.35); }

/* ── Streamlit tab overrides ── */
[data-testid="stTabs"] { margin-top: 0; }
button[data-baseweb="tab"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.5) !important;
    padding: 8px 20px !important;
    border-radius: 8px 8px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: white !important;
    border-bottom: 2px solid #e94560 !important;
}
[data-testid="stTabsContent"] { padding-top: 1.5rem; }

/* ── Movie cards ── */
.movie-card {
    background: #1a1a2e;
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.55rem;
    color: white;
    transition: border-color 0.15s;
}
.movie-card:hover { border-color: #e94560; }
.card-rank  { font-size: 0.68rem; color: #e94560; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.card-title { font-size: 0.95rem; font-weight: 600; margin: 3px 0; }
.card-meta  { font-size: 0.72rem; color: rgba(255,255,255,0.4); margin-top: 2px; }

/* ── Genre badges ── */
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 0.67rem;
    font-weight: 500;
    margin: 2px 2px 0 0;
}

/* ── KPI strip ── */
.kpi-strip {
    display: flex;
    gap: 10px;
    margin: 1rem 0 1.2rem;
}
.kpi {
    background: #1a1a2e;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    flex: 1;
}
.kpi .kval { font-size: 1.3rem; font-weight: 700; color: #e94560; }
.kpi .klbl { font-size: 0.68rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; }

/* ── Section title ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    border-left: 3px solid #e94560;
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem;
    border-radius: 0;
}

/* ── Algo pill ── */
.algo-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(90deg, #e94560, #c62a47);
    color: white;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.73rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* ── Comparison cards ── */
.algo-group-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.35);
    margin: 1rem 0 0.4rem;
}
.compare-card {
    background: #12122a;
    border: 1px solid #1e1e44;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.4rem;
}
.compare-card .algo-name { font-size: 0.78rem; font-weight: 700; color: white; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
.compare-card .rec-item  { font-size: 0.75rem; color: rgba(255,255,255,0.55); padding: 3px 0; border-bottom: 1px solid #1e1e44; }
.compare-card .rec-item:last-child { border-bottom: none; }

/* ── Stat cards (dataset page) ── */
.stat-card {
    background: #1a1a2e;
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    color: white;
}
.stat-card .sval { font-size: 1.8rem; font-weight: 700; color: #e94560; }
.stat-card .slbl { font-size: 0.72rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 3px; }

/* ── Metric overrides ── */
[data-testid="stMetricValue"] { color: #e94560 !important; }

/* ── Label colours ── */
.stSelectbox label, .stSlider label, .stTextInput label, .stMultiSelect label {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}

/* ── Dark background ── */
.stApp { background: #0d0d1a; }
[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Data download ─────────────────────────────────────────────────────────────
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DIR = Path("data/raw")

def ensure_data():
    if (RAW_DIR / "ratings.csv").exists() and (RAW_DIR / "movies.csv").exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading MovieLens dataset (first run only)..."):
        with urllib.request.urlopen(DATA_URL) as resp:
            zipped = zipfile.ZipFile(io.BytesIO(resp.read()))
        for name in zipped.namelist():
            fname = Path(name).name
            if fname in ("ratings.csv", "movies.csv", "tags.csv"):
                (RAW_DIR / fname).write_bytes(zipped.read(name))

ensure_data()

# ── Imports ───────────────────────────────────────────────────────────────────
from src import config
from src.data_loading import load_ratings, load_items, train_test_split_ratings
from src.baselines import MostPopularRecommender, HighestAverageRatingRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import (
    precision_at_k, recall_at_k, ndcg_at_k,
    mean_reciprocal_rank, hit_rate_at_k,
    catalog_coverage, novelty_at_k,
)

# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    ratings = load_ratings()
    items   = load_items()
    tags    = pd.read_csv(RAW_DIR / "tags.csv")
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    return ratings, items, tags, train, test


@st.cache_resource(show_spinner="Training models, this takes about 2 minutes on first load...")
def train_models(_train, _items, _tags):
    models = {
        "Most Popular":              MostPopularRecommender(),
        "Highest Rated":             HighestAverageRatingRecommender(min_ratings=20),
        "Content-Based (Genres)":    ContentBasedRecommender(use_tags=False),
        "Content-Based (+Tags)":     ContentBasedRecommender(use_tags=True),
        "Item-Item CF":              ItemItemCollaborativeFiltering(k=20),
        "User-User CF":              UserUserCollaborativeFiltering(k=20),
        "Matrix Factorisation":      MatrixFactorizationRecommender(n_factors=50, n_epochs=20),
    }
    for name, model in models.items():
        if "Content" in name:
            model.fit(_train, _items, _tags)
        elif name in ("Item-Item CF", "User-User CF", "Matrix Factorisation"):
            model.fit(_train)
        else:
            model.fit(_train, _items)
    return models


ratings, items, tags, train, test = load_data()
models = train_models(train, items, tags)

# ── Helpers ───────────────────────────────────────────────────────────────────
title_map    = items.set_index(config.ITEM_COL)[config.TITLE_COL].to_dict()
genre_map    = items.set_index(config.ITEM_COL)[config.GENRES_COL].to_dict()
all_users    = sorted(train[config.USER_COL].unique().tolist())
item_pop     = train.groupby(config.ITEM_COL)[config.RATING_COL].count().to_dict()
n_users_tr   = train[config.USER_COL].nunique()
all_item_ids = items[config.ITEM_COL].tolist()

ALL_GENRES = sorted(
    {g for gs in items[config.GENRES_COL].dropna() for g in gs.split("|")
     if g != "(no genres listed)"}
)

GENRE_COLORS = {
    "Action":"#e94560","Adventure":"#f0a500","Animation":"#00b4d8",
    "Comedy":"#f72585","Crime":"#7209b7","Documentary":"#4cc9f0",
    "Drama":"#4361ee","Fantasy":"#3a0ca3","Film-Noir":"#555",
    "Horror":"#e63946","Musical":"#f77f00","Mystery":"#560bad",
    "Romance":"#ff006e","Sci-Fi":"#06d6a0","Thriller":"#118ab2",
    "War":"#4a6fa5","Western":"#8d6a9f",
}

ALGO_GROUPS = {
    "Baseline":    ["Most Popular", "Highest Rated"],
    "Content":     ["Content-Based (Genres)", "Content-Based (+Tags)"],
    "Collaborative": ["Item-Item CF", "User-User CF"],
    "Matrix":      ["Matrix Factorisation"],
}

ALGO_ICONS = {
    "Most Popular":           "ti-trending-up",
    "Highest Rated":          "ti-star",
    "Content-Based (Genres)": "ti-tags",
    "Content-Based (+Tags)":  "ti-tag",
    "Item-Item CF":           "ti-arrows-exchange",
    "User-User CF":           "ti-users",
    "Matrix Factorisation":   "ti-math-function",
}

GROUP_COLORS = {
    "Baseline":      "#f0a500",
    "Content":       "#4361ee",
    "Collaborative": "#06d6a0",
    "Matrix":        "#e94560",
}

def genre_badges(genres_str):
    if not genres_str or genres_str == "(no genres listed)":
        return ""
    html = ""
    for g in genres_str.split("|")[:4]:
        c = GENRE_COLORS.get(g, "#0f3460")
        html += f'<span class="badge" style="background:{c}22;color:{c};border:1px solid {c}44">{g}</span>'
    return html

def star_icons(rating):
    filled = int(rating)
    half   = 1 if (rating - filled) >= 0.5 else 0
    empty  = 5 - filled - half
    s  = '<i class="ti ti-star-filled" style="color:#e94560;font-size:12px"></i>' * filled
    s += '<i class="ti ti-star-half-filled" style="color:#e94560;font-size:12px"></i>' * half
    s += '<i class="ti ti-star" style="color:#444;font-size:12px"></i>' * empty
    return s

def movie_card(rank, item_id, why=None):
    title  = title_map.get(item_id, f"Movie {item_id}")
    genres = genre_map.get(item_id, "")
    pop    = item_pop.get(item_id, 0)
    why_html = ""
    if why:
        why_html = f'<div class="card-meta" style="margin-top:5px;color:#06d6a0;font-style:italic">{why}</div>'
    st.markdown(f"""
    <div class="movie-card">
        <div class="card-rank">#{rank}</div>
        <div class="card-title">{title}</div>
        <div style="margin:4px 0">{genre_badges(genres)}</div>
        <div class="card-meta">{pop:,} ratings</div>
        {why_html}
    </div>
    """, unsafe_allow_html=True)

def filter_by_genre(item_ids, genre):
    """Keep only items that belong to the selected genre."""
    if not genre or genre == "All genres":
        return item_ids
    return [i for i in item_ids if genre in genre_map.get(i, "").split("|")]

# ── Global header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- board body -->
    <rect x="2" y="10" width="32" height="22" rx="2" fill="#1a1a2e" stroke="#e94560" stroke-width="1.5"/>
    <!-- clap top (white stripes panel) -->
    <polygon points="2,10 34,10 30,4 6,4" fill="#f0f0f0"/>
    <!-- diagonal black stripes on clap -->
    <clipPath id="clap-clip"><polygon points="2,10 34,10 30,4 6,4"/></clipPath>
    <g clip-path="url(#clap-clip)">
      <rect x="6"  y="3" width="4" height="9" fill="#222" transform="skewX(-15)"/>
      <rect x="14" y="3" width="4" height="9" fill="#222" transform="skewX(-15)"/>
      <rect x="22" y="3" width="4" height="9" fill="#222" transform="skewX(-15)"/>
    </g>
    <!-- hinge line -->
    <line x1="2" y1="10" x2="34" y2="10" stroke="#e94560" stroke-width="1.5"/>
    <!-- film holes left -->
    <rect x="5"  y="15" width="4" height="3" rx="1" fill="#0d0d1a"/>
    <rect x="5"  y="22" width="4" height="3" rx="1" fill="#0d0d1a"/>
    <rect x="5"  y="29" width="4" height="2" rx="1" fill="#0d0d1a"/>
    <!-- film holes right -->
    <rect x="27" y="15" width="4" height="3" rx="1" fill="#0d0d1a"/>
    <rect x="27" y="22" width="4" height="3" rx="1" fill="#0d0d1a"/>
    <rect x="27" y="29" width="4" height="2" rx="1" fill="#0d0d1a"/>
  </svg>
  <div>
    <div class="logo-name">Cinemetric</div>
    <div class="logo-sub">Recommender Systems Project · MovieLens Latest Small</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top navigation tabs ───────────────────────────────────────────────────────
tab_recs, tab_data, tab_compare, tab_bias, tab_how = st.tabs([
    "Recommendations", "Explore Dataset", "Model Comparison", "Popularity Bias", "How It Works"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_recs:

    # ── Controls row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1])
    with c1:
        user_id    = st.selectbox("User", all_users)
    with c2:
        model_name = st.selectbox("Algorithm", list(models.keys()))
    with c3:
        genre_filter = st.selectbox("Filter by genre", ["All genres"] + ALL_GENRES)
    with c4:
        n_recs = st.slider("Top N", 5, 20, 10)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    user_train = train[train[config.USER_COL] == user_id]
    avg_rating = user_train[config.RATING_COL].mean() if len(user_train) else 0
    fav_genre  = (
        user_train.merge(items, on=config.ITEM_COL)[config.GENRES_COL]
        .str.split("|").explode().value_counts().index[0]
        if len(user_train) > 0 else "N/A"
    )
    genre_count = len([i for i in user_train[config.ITEM_COL]
                       if fav_genre in genre_map.get(i, "").split("|")])

    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi">
            <div class="klbl">User</div>
            <div class="kval">#{user_id}</div>
        </div>
        <div class="kpi">
            <div class="klbl">Movies rated</div>
            <div class="kval">{len(user_train):,}</div>
        </div>
        <div class="kpi">
            <div class="klbl">Avg rating</div>
            <div class="kval">{avg_rating:.1f} <span style="font-size:1rem;color:rgba(255,255,255,0.3)">/ 5</span></div>
        </div>
        <div class="kpi">
            <div class="klbl">Top genre</div>
            <div class="kval" style="font-size:1rem">{fav_genre}</div>
        </div>
        <div class="kpi">
            <div class="klbl">{fav_genre} films rated</div>
            <div class="kval">{genre_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recommendations + history ─────────────────────────────────────────────
    icon = ALGO_ICONS.get(model_name, "ti-sparkles")
    st.markdown(f'<div class="algo-pill"><i class="ti {icon}"></i> {model_name}</div>', unsafe_allow_html=True)

    with st.spinner("Generating recommendations..."):
        raw_recs = models[model_name].recommend(user_id, train, n=50)
        recs     = filter_by_genre(raw_recs, genre_filter)[:n_recs]

    filter_note = f" in **{genre_filter}**" if genre_filter != "All genres" else ""
    st.markdown(f'<div class="section-title">Top {n_recs} picks for User {user_id}{" · " + genre_filter if genre_filter != "All genres" else ""}</div>', unsafe_allow_html=True)

    if not recs:
        st.warning(f"No recommendations found{filter_note}. Try a different genre or algorithm.")
    else:
        rec_col, hist_col = st.columns([3, 2])

        with rec_col:
            is_content = "Content" in model_name
            for rank, item_id in enumerate(recs, 1):
                why = None
                if is_content:
                    reasons = models[model_name].explain_recommendation(
                        user_id, item_id, user_train, n=2
                    )
                    if reasons:
                        parts = [
                            f"{title_map.get(rid, rid)[:22]} ({r:.1f}★)"
                            for rid, r, _ in reasons
                        ]
                        why = "Because you liked: " + " · ".join(parts)
                movie_card(rank, item_id, why=why)

        with hist_col:
            st.markdown('<div class="section-title">Watch history</div>', unsafe_allow_html=True)
            history_df = user_train.sort_values(config.RATING_COL, ascending=False)
            if genre_filter != "All genres":
                history_df = history_df[
                    history_df[config.ITEM_COL].map(
                        lambda i: genre_filter in genre_map.get(i, "").split("|")
                    )
                ]
            for _, row in history_df.head(8).iterrows():
                iid    = row[config.ITEM_COL]
                title  = title_map.get(iid, iid)
                genres = genre_map.get(iid, "")
                stars  = star_icons(row[config.RATING_COL])
                st.markdown(f"""
                <div class="movie-card">
                    <div class="card-title" style="font-size:0.88rem">{title}</div>
                    <div style="margin:3px 0">{stars}
                        <span style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-left:4px">{row[config.RATING_COL]}</span>
                    </div>
                    <div>{genre_badges(genres)}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Algorithm comparison ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Algorithm comparison: same user, all models</div>', unsafe_allow_html=True)
    st.caption(
        f"What would each algorithm recommend to User {user_id}? "
        "Grouped by approach so you can see how the strategy affects the output."
    )

    group_cols = st.columns(len(ALGO_GROUPS))
    for col, (group_name, algo_names) in zip(group_cols, ALGO_GROUPS.items()):
        color = GROUP_COLORS[group_name]
        with col:
            st.markdown(
                f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:1px;color:{color};border-bottom:2px solid {color};'
                f'padding-bottom:4px;margin-bottom:8px">{group_name}</div>',
                unsafe_allow_html=True,
            )
            for algo in algo_names:
                top5 = models[algo].recommend(user_id, train, n=5)
                top5 = filter_by_genre(top5, genre_filter)[:5]
                icon = ALGO_ICONS.get(algo, "ti-sparkles")
                items_html = "".join(
                    f'<div class="rec-item">{title_map.get(r, r)[:35]}{"…" if len(title_map.get(r,""))>35 else ""}</div>'
                    for r in top5
                ) or '<div class="rec-item" style="opacity:0.4">No results for this genre</div>'
                st.markdown(f"""
                <div class="compare-card" style="border-top:2px solid {color}22">
                    <div class="algo-name">
                        <i class="ti {icon}" style="color:{color};font-size:14px"></i>
                        {algo}
                    </div>
                    {items_html}
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Explore Dataset
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:

    st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)

    n_u      = ratings[config.USER_COL].nunique()
    n_i      = ratings[config.ITEM_COL].nunique()
    n_r      = len(ratings)
    sparsity = 1 - n_r / (n_u * n_i)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{n_u:,}", f"{n_i:,}", f"{n_r:,}", f"{sparsity:.1%}"],
        ["Users", "Movies", "Ratings", "Sparsity"],
    ):
        col.markdown(f"""
        <div class="stat-card">
            <div class="sval">{val}</div>
            <div class="slbl">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Genre filter for dataset exploration ──────────────────────────────────
    explore_genre = st.selectbox(
        "Filter movies by genre",
        ["All genres"] + ALL_GENRES,
        key="explore_genre",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Rating distribution</div>', unsafe_allow_html=True)
        dist = ratings[config.RATING_COL].value_counts().sort_index().reset_index()
        dist.columns = ["Rating", "Count"]
        st.bar_chart(dist.set_index("Rating"), color="#e94560")

    with col_b:
        st.markdown('<div class="section-title">Most rated movies</div>', unsafe_allow_html=True)
        top_movies = (
            ratings.groupby(config.ITEM_COL)[config.RATING_COL]
            .agg(["count", "mean"])
            .reset_index()
            .merge(items[[config.ITEM_COL, config.TITLE_COL, config.GENRES_COL]], on=config.ITEM_COL)
        )
        if explore_genre != "All genres":
            top_movies = top_movies[
                top_movies[config.GENRES_COL].str.contains(explore_genre, na=False)
            ]
        top_movies = top_movies.sort_values("count", ascending=False).head(10)
        for _, row in top_movies.iterrows():
            st.markdown(f"""
            <div class="movie-card">
                <div class="card-title" style="font-size:0.88rem">{row[config.TITLE_COL]}</div>
                <div>{genre_badges(row[config.GENRES_COL])}</div>
                <div class="card-meta">{int(row['count']):,} ratings · avg {row['mean']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Genre breakdown</div>', unsafe_allow_html=True)
    genre_counts = (
        items[config.GENRES_COL].str.split("|").explode()
        .value_counts().reset_index()
    )
    genre_counts.columns = ["Genre", "Count"]
    genre_counts = genre_counts[genre_counts["Genre"] != "(no genres listed)"]
    st.bar_chart(genre_counts.set_index("Genre").head(18), color="#4361ee")

    st.markdown("---")
    st.markdown('<div class="section-title">Search movies</div>', unsafe_allow_html=True)
    query = st.text_input("Search by title...", placeholder="e.g. Matrix")
    if query:
        matches = items[items[config.TITLE_COL].str.contains(query, case=False, na=False)]
        if explore_genre != "All genres":
            matches = matches[matches[config.GENRES_COL].str.contains(explore_genre, na=False)]
        matches = matches.head(8)
        if matches.empty:
            st.info("No movies found.")
        else:
            for _, row in matches.iterrows():
                iid = row[config.ITEM_COL]
                pop = item_pop.get(iid, 0)
                avg = ratings[ratings[config.ITEM_COL] == iid][config.RATING_COL].mean()
                st.markdown(f"""
                <div class="movie-card">
                    <div class="card-title">{row[config.TITLE_COL]}</div>
                    <div>{genre_badges(row[config.GENRES_COL])}</div>
                    <div class="card-meta">{pop:,} ratings · avg {avg:.2f}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:

    results_path = Path("results/metrics.csv")
    if not results_path.exists():
        st.info("Run `python3 main.py` first to generate `results/metrics.csv`.")
    else:
        df = pd.read_csv(results_path, index_col=0)

        st.markdown('<div class="section-title">Rank models by metric</div>', unsafe_allow_html=True)
        metric = st.selectbox(
            "Metric",
            ["ndcg", "precision", "recall", "mrr", "hit_rate", "coverage", "novelty"],
            format_func=lambda x: {
                "ndcg":"NDCG@10","precision":"Precision@10","recall":"Recall@10",
                "mrr":"MRR","hit_rate":"Hit Rate@10","coverage":"Catalog Coverage","novelty":"Novelty",
            }[x],
        )

        ranked   = df[metric].sort_values(ascending=False)
        best_val = ranked.max()
        for mdl, val in ranked.items():
            pct    = val / best_val if best_val > 0 else 0
            is_top = val == best_val
            trophy = '<i class="ti ti-trophy" style="color:#f0a500;font-size:13px;margin-left:4px"></i>' if is_top else ""
            st.markdown(f"""
            <div style="margin-bottom:0.6rem">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;align-items:center">
                    <span style="color:white;font-size:0.88rem;font-weight:{'700' if is_top else '400'}">{mdl}{trophy}</span>
                    <span style="color:#e94560;font-weight:600;font-size:0.88rem">{val:.4f}</span>
                </div>
                <div style="background:#1a1a2e;border-radius:4px;height:7px">
                    <div style="background:{'#e94560' if is_top else '#0f3460'};width:{pct*100:.1f}%;height:7px;border-radius:4px;transition:width 0.3s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Full results table</div>', unsafe_allow_html=True)
        display_df = df.copy().round(4)
        display_df.columns = [c.upper() for c in display_df.columns]
        st.dataframe(
            display_df.style.highlight_max(axis=0, color="#1a3a1a").highlight_min(axis=0, color="#3a1a1a"),
            use_container_width=True,
        )

        st.markdown("---")
        cx, cy = st.columns(2)
        with cx:
            st.markdown('<div class="section-title">Accuracy (NDCG@10)</div>', unsafe_allow_html=True)
            st.bar_chart(df[["ndcg"]].sort_values("ndcg"), color="#e94560")
        with cy:
            st.markdown('<div class="section-title">Catalog coverage</div>', unsafe_allow_html=True)
            st.bar_chart(df[["coverage"]].sort_values("coverage"), color="#4361ee")

        st.markdown("---")
        st.markdown('<div class="section-title">Key takeaways</div>', unsafe_allow_html=True)

        # Derive takeaways from actual metric numbers
        personalised = [m for m in df.index if m not in ("random", "most_popular", "highest_avg")]
        best_overall      = df["ndcg"].idxmax()
        best_personalised = df.loc[personalised, "ndcg"].idxmax() if personalised else best_overall
        best_coverage     = df["coverage"].idxmax()
        worst_accuracy    = df["ndcg"].idxmin()
        cb_tags_better    = (
            "content_based_tags" in df.index and "content_based" in df.index and
            df.loc["content_based_tags", "precision"] > df.loc["content_based", "precision"]
        )

        name_fmt = lambda n: n.replace("_", " ").title()

        data_takeaways = [
            (
                f"{name_fmt(best_overall)} wins on accuracy (NDCG@10 = {df.loc[best_overall,'ndcg']:.4f})",
                "Non-personalised baselines are a known hard benchmark: popular items are popular for a reason. "
                "This is the accuracy-vs-personalisation trade-off at the heart of recommender systems research."
            ),
            (
                f"{name_fmt(best_personalised)} is the best personalised model (NDCG@10 = {df.loc[best_personalised,'ndcg']:.4f})",
                "Latent factor models handle the 98.3% sparsity better than neighbourhood methods by learning "
                "dense representations. The gap over User-User CF is "
                f"{df.loc[best_personalised,'ndcg']/max(df.loc['user_user_cf','ndcg'],1e-9):.1f}× on NDCG."
                if "user_user_cf" in df.index else
                "Latent factor models handle sparsity better than neighbourhood-based CF."
            ),
            (
                f"{'Tags improve' if cb_tags_better else 'Tags marginally affect'} content-based precision",
                f"Content-Based (+Tags): Precision = {df.loc['content_based_tags','precision']:.4f} vs "
                f"Genres only: {df.loc['content_based','precision']:.4f}. "
                + ("User tags add personalised vocabulary that genre labels alone cannot capture."
                   if cb_tags_better else
                   "On this dataset, genre features already capture most of the signal.")
                if "content_based_tags" in df.index and "content_based" in df.index else
                "Content-based models leverage item metadata to recommend similar genres."
            ),
            (
                f"Accuracy ≠ Coverage: {name_fmt(best_overall)} covers {df.loc[best_overall,'coverage']:.1%} vs "
                f"{name_fmt(best_coverage)} covers {df.loc[best_coverage,'coverage']:.1%}",
                "The model with the best accuracy recommends a tiny slice of the catalog to everyone. "
                f"{name_fmt(best_coverage)} surfaces far more of the long tail, essential for content discovery "
                "and avoiding the popularity feedback loop."
            ),
        ]

        for title, body in data_takeaways:
            st.markdown(f"""
            <div class="movie-card">
                <div class="card-title">{title}</div>
                <div class="card-meta" style="margin-top:4px;line-height:1.6">{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: How It Works
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Popularity Bias
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bias:
    import matplotlib.pyplot as plt
    import matplotlib

    st.markdown('<div class="section-title">What is popularity bias?</div>', unsafe_allow_html=True)
    st.markdown("""
    Recommender systems trained on interaction data tend to recommend **popular items far
    more often than niche ones** even when personalised recommendations should surface
    hidden gems. This creates a feedback loop: popular items get more exposure → more ratings
    → appear even more popular. The long tail of the catalog is systematically ignored.
    """)

    # ── Build popularity tiers ────────────────────────────────────────────────
    pop_series = train.groupby(config.ITEM_COL)[config.RATING_COL].count().sort_values(ascending=False)
    pop_df_app = pop_series.reset_index()
    pop_df_app.columns = [config.ITEM_COL, 'n_ratings']
    cumulative_app = pop_df_app['n_ratings'].cumsum() / pop_df_app['n_ratings'].sum()
    head_cut_app  = int((cumulative_app <= 0.20).sum())
    torso_cut_app = int((cumulative_app <= 0.50).sum())

    def assign_tier_app(idx):
        if idx < head_cut_app:  return 'Head'
        if idx < torso_cut_app: return 'Torso'
        return 'Tail'

    pop_df_app['tier'] = [assign_tier_app(i) for i in range(len(pop_df_app))]
    item_to_tier_app   = dict(zip(pop_df_app[config.ITEM_COL], pop_df_app['tier']))
    TIER_COLORS_APP    = {'Head': '#e94560', 'Torso': '#f0a500', 'Tail': '#4361ee'}

    tier_counts_app = pop_df_app['tier'].value_counts()
    c1, c2, c3 = st.columns(3)
    for col, tier in zip([c1, c2, c3], ['Head', 'Torso', 'Tail']):
        n = tier_counts_app.get(tier, 0)
        col.markdown(f"""
        <div class="stat-card">
            <div class="sval" style="color:{TIER_COLORS_APP[tier]}">{n:,}</div>
            <div class="slbl">{tier} ({n/len(pop_df_app):.0%} of catalog)</div>
        </div>""", unsafe_allow_html=True)

    st.caption("Head = top 20% of rating volume · Torso = next 30% · Tail = remaining 50%")

    # ── Sample recommendations from all models ────────────────────────────────
    N_BIAS_USERS   = 20
    rng_bias       = np.random.default_rng(99)
    bias_users     = rng_bias.choice(train[config.USER_COL].unique(),
                                     size=min(N_BIAS_USERS, train[config.USER_COL].nunique()),
                                     replace=False)
    # Pre-group so per-user lookup is O(1)
    train_by_user  = {uid: grp for uid, grp in train.groupby(config.USER_COL)}

    if 'bias_recs_v3' not in st.session_state:
        with st.spinner("Analysing popularity bias (sampling 20 users × 7 models)..."):
            recs_by_model = {}
            for name, model in models.items():
                recs = []
                for uid in bias_users:
                    user_train_df = train_by_user.get(uid)
                    if user_train_df is None:
                        continue
                    try:
                        r = model.recommend(uid, user_train_df, n=10)
                        recs.extend(r)
                    except Exception:
                        pass
                recs_by_model[name] = recs
        st.session_state['bias_recs_v3'] = recs_by_model
    else:
        recs_by_model = st.session_state['bias_recs_v3']

    # Tier fractions
    tier_frac_app = {}
    for name, recs in recs_by_model.items():
        tiers = [item_to_tier_app.get(mid, 'Tail') for mid in recs]
        total = len(tiers) or 1
        tier_frac_app[name] = {t: tiers.count(t)/total for t in ['Head','Torso','Tail']}

    frac_df_app = pd.DataFrame(tier_frac_app).T

    # Avg popularity
    avg_pop_app = {}
    for name, recs in recs_by_model.items():
        pops = [item_pop.get(mid, 0) for mid in recs]
        avg_pop_app[name] = float(np.mean(pops)) if pops else 0

    # Gini
    def gini_app(values):
        arr = np.sort(np.array(values, dtype=float))
        n   = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return (2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n

    gini_app_scores = {}
    all_item_ids_arr = items[config.ITEM_COL].values
    for name, recs in recs_by_model.items():
        freq = pd.Series(recs).value_counts().reindex(all_item_ids_arr, fill_value=0)
        gini_app_scores[name] = gini_app(freq.values)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Tier distribution per algorithm</div>', unsafe_allow_html=True)
    matplotlib.rcParams['figure.facecolor'] = '#0d0d1a'
    matplotlib.rcParams['axes.facecolor']   = '#1a1a2e'
    matplotlib.rcParams['axes.edgecolor']   = '#0f3460'
    matplotlib.rcParams['text.color']       = 'white'
    matplotlib.rcParams['axes.labelcolor']  = 'white'
    matplotlib.rcParams['xtick.color']      = '#aaaaaa'
    matplotlib.rcParams['ytick.color']      = '#aaaaaa'

    fig_tier, ax_tier = plt.subplots(figsize=(10, 4))
    fig_tier.patch.set_facecolor('#0d0d1a')
    ax_tier.set_facecolor('#1a1a2e')
    bottom = np.zeros(len(frac_df_app))
    for tier, color in TIER_COLORS_APP.items():
        vals = frac_df_app[tier].values
        bars = ax_tier.bar(frac_df_app.index, vals, bottom=bottom, color=color,
                           label=tier, width=0.55)
        for bar, v in zip(bars, vals):
            if v > 0.07:
                ax_tier.text(bar.get_x() + bar.get_width()/2,
                             bar.get_y() + bar.get_height()/2,
                             f'{v:.0%}', ha='center', va='center',
                             fontsize=8, color='white', fontweight='bold')
        bottom += vals
    ax_tier.set_ylabel('Fraction of recommendations', color='white')
    ax_tier.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax_tier.tick_params(axis='x', rotation=20)
    ax_tier.legend(loc='upper right', framealpha=0.2)
    ax_tier.set_ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig_tier)
    plt.close(fig_tier)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="section-title">Average popularity of recommendations</div>', unsafe_allow_html=True)
        avg_s = pd.Series(avg_pop_app).sort_values(ascending=True)
        catalog_mean = float(pop_series.mean())
        fig_avg, ax_avg = plt.subplots(figsize=(6, 3.5))
        fig_avg.patch.set_facecolor('#0d0d1a')
        ax_avg.set_facecolor('#1a1a2e')
        colors_avg = ['#e94560' if v > catalog_mean else '#4361ee' for v in avg_s.values]
        ax_avg.barh(avg_s.index, avg_s.values, color=colors_avg, height=0.55)
        ax_avg.axvline(catalog_mean, color='white', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Catalog mean ({catalog_mean:.0f})')
        ax_avg.set_xlabel('Mean ratings of recommended items')
        ax_avg.legend(fontsize=7, framealpha=0.2)
        ax_avg.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig_avg)
        plt.close(fig_avg)

    with cb:
        st.markdown('<div class="section-title">Recommendation inequality (Gini)</div>', unsafe_allow_html=True)
        gini_s_app = pd.Series(gini_app_scores).sort_values(ascending=True)
        fig_gini, ax_gini = plt.subplots(figsize=(6, 3.5))
        fig_gini.patch.set_facecolor('#0d0d1a')
        ax_gini.set_facecolor('#1a1a2e')
        ax_gini.barh(gini_s_app.index, gini_s_app.values, color='#06d6a0', height=0.55)
        ax_gini.set_xlabel('Gini coefficient (lower = more diverse)')
        ax_gini.set_xlim(0, 1)
        ax_gini.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig_gini)
        plt.close(fig_gini)

    st.markdown("---")
    st.markdown('<div class="section-title">Key takeaways</div>', unsafe_allow_html=True)
    takeaways = [
        ("Most Popular always wins on head bias",
         "By definition it only recommends frequently-rated films. High precision, zero novelty."),
        ("Collaborative filtering amplifies popularity",
         "Item-item CF similarity is dominated by co-occurrence, which skews toward blockbusters."),
        ("Content-based is relatively fairer",
         "Operating on genre text rather than interaction counts, it can surface niche films with matching features."),
        ("Matrix factorisation sits in the middle",
         "Latent factors capture popularity as a dimension but also encode genre taste, giving moderate diversity."),
        ("Random is unbiased but useless",
         "The only algorithm with a truly flat tier distribution, because it knows nothing about the user."),
    ]
    for title, body in takeaways:
        st.markdown(f"""
        <div class="movie-card">
            <div class="card-title">{title}</div>
            <div class="card-meta" style="margin-top:4px;line-height:1.7">{body}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: How It Works
# ═══════════════════════════════════════════════════════════════════════════════
with tab_how:

    st.markdown('<div class="section-title">Seven algorithms, one dataset</div>', unsafe_allow_html=True)
    st.caption("Each algorithm uses a fundamentally different strategy to decide what to recommend.")

    algo_info = [
        ("Most Popular",           "ti-trending-up", "Baseline",      "#f0a500",
         "Recommend what everyone watches",
         "Ranks movies by total number of ratings. No personalisation, same list for everyone. Surprisingly hard to beat.",
         "Simple, fast, no cold-start problem", "Zero personalisation, popularity bias"),
        ("Highest Rated",          "ti-star",        "Baseline",      "#f0a500",
         "Recommend what critics love",
         "Ranks by average rating, filtered to movies with at least 20 ratings to avoid obscure films with a single perfect score.",
         "Surfaces quality over popularity", "Still not personalised, misses niche tastes"),
        ("Content-Based (Genres)", "ti-tags",        "Content",       "#4361ee",
         "Match movies to your taste profile",
         "Converts genres to TF-IDF vectors. Builds a user profile as a weighted sum of rated movie vectors, centered by the user's mean rating. Recommends by cosine similarity.",
         "Personalised, no other users needed", "Limited to metadata features only"),
        ("Content-Based (+Tags)",  "ti-tag",         "Content",       "#4361ee",
         "Genres plus the words fans use",
         "Same as above but appends user-generated tags (e.g. 'twist ending') to the genre text. Tags capture nuance that genre labels miss.",
         "Richer representation, measurably better", "Tags are sparse, many movies have none"),
        ("Item-Item CF",           "ti-arrows-exchange","Collaborative","#06d6a0",
         "Because you liked X, try Y",
         "Cosine similarity between all movie pairs based on shared raters. Predicts score via weighted average of ratings on similar movies.",
         "Stable, no metadata needed", "Slow for large catalogs, hurt by sparsity"),
        ("User-User CF",           "ti-users",       "Collaborative", "#06d6a0",
         "People like you also liked...",
         "Finds similar users via cosine similarity on rating vectors. Predicts ratings from neighbour scores.",
         "Highly personalised", "Very sensitive to sparsity"),
        ("Matrix Factorisation",   "ti-math-function","Matrix",       "#e94560",
         "Learning hidden taste dimensions",
         "SGD model: r̂(u,i) = μ + b_u + b_i + p_u · q_i. Biases capture systematic tendencies; latent factors capture abstract taste patterns.",
         "Best personalised accuracy, handles sparsity", "Black box, hard to explain individual predictions"),
    ]

    for name, icon, group, color, tagline, how, strength, weakness in algo_info:
        with st.expander(f"{name}: {tagline}"):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
                f'<i class="ti {icon}" style="font-size:1.2rem;color:{color}"></i>'
                f'<span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:1px;color:{color}">{group}</span></div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**How it works**")
                st.write(how)
            with c2:
                st.markdown("**Strength**")
                st.success(strength)
                st.markdown("**Weakness**")
                st.error(weakness)

    st.markdown("---")
    st.markdown('<div class="section-title" style="margin-top:0">Evaluation protocol</div>', unsafe_allow_html=True)
    for title, body in [
        ("Temporal train / test split",
         "Each user's most recent 20% of ratings are held out. This mirrors real deployment: train on history, predict what comes next. A random split would leak future data into training."),
        ("Metrics at K = 10",
         "Precision: fraction of top-10 that are relevant. Recall: fraction of all relevant items found. NDCG: precision with position weighting. MRR: rank of first hit. Coverage: share of catalog ever recommended. Novelty: mean self-information of recommended items."),
    ]:
        st.markdown(f"""
        <div class="movie-card">
            <div class="card-title">{title}</div>
            <div class="card-meta" style="margin-top:4px;line-height:1.7">{body}</div>
        </div>
        """, unsafe_allow_html=True)
