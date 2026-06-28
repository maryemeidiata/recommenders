# Cinemetric: A Movie Recommender System

A full recommender system prototype built on the **MovieLens Latest Small** dataset (610 users, 9,724 movies, 100,836 ratings).

**Live app:** [https://recommendersapp.streamlit.app/](https://recommendersapp.streamlit.app/)

---

## Algorithms implemented

| Algorithm | Type |
|---|---|
| Most Popular | Non-personalised baseline |
| Highest Average Rating | Non-personalised baseline |
| Random | Non-personalised baseline |
| Content-Based (Genres) | Content-based filtering |
| Content-Based (Genres + Tags) | Content-based filtering |
| Item-Item Collaborative Filtering | Memory-based CF |
| User-User Collaborative Filtering | Memory-based CF |
| Matrix Factorisation (SGD) | Model-based |

---

## Evaluation results (K=10, 200 users)

| Model | Precision | Recall | NDCG | Coverage | Novelty |
|---|---|---|---|---|---|
| Most Popular | 0.071 | 0.053 | 0.090 | 0.79% | 1.53 |
| Matrix Fact | 0.041 | 0.023 | 0.043 | 1.82% | 3.33 |
| Highest Avg | 0.026 | 0.015 | 0.035 | 0.34% | 3.68 |
| CB + Tags | 0.011 | 0.007 | 0.012 | 11.65% | 7.26 |
| Content-Based | 0.008 | 0.006 | 0.012 | 11.18% | 6.95 |
| Item-Item CF | 0.001 | 0.000 | 0.001 | 9.21% | 8.78 |
| Random | 0.006 | 0.001 | 0.006 | 7.76% | 7.23 |
| User-User CF | 0.000 | 0.000 | 0.000 | 1.44% | 9.13 |

Evaluation uses a temporal train/test split and a relevance threshold of 3.5 stars.

---

## Project structure

```
src/
  baselines.py              Non-personalised recommenders
  content_based.py          TF-IDF content-based filtering
  collaborative_filtering.py  Item-item and user-user CF (mean-centered)
  matrix_factorization.py   SGD matrix factorisation
  evaluation.py             Precision, Recall, NDCG, MRR, Hit Rate, Coverage, Novelty, Diversity
  data_loading.py           Data loading and temporal train/test split
  config.py                 Shared constants

notebooks/
  01_eda.ipynb              Exploratory data analysis
  02_popularity_bias.ipynb  Popularity bias analysis (Head/Torso/Tail, Gini)

app.py                      Streamlit web application
main.py                     Evaluation pipeline, saves results/metrics.csv
results/metrics.csv         Evaluation results for all models
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Run evaluation

```bash
python main.py
```

## Run app locally

```bash
streamlit run app.py
```
