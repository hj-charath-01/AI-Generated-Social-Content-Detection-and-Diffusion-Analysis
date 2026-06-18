# AI-Generated Social Content Detection and Diffusion Analysis

A modular, production-style research system that:
1. **Detects** whether a social media post is human-written or LLM-generated
2. **Analyzes** how AI-generated content diffuses through social networks vs. human content

---

## Quick Start

```bash
# 1. Install PyTorch (CPU-only — no GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install all other dependencies
pip install -r requirements.txt

# 3. Generate synthetic demo dataset
python main.py generate --n-posts 2000

# 3. Train all models
python main.py train --data data/synthetic_posts.csv

# 4. Run inference on new data
python main.py infer data/synthetic_posts.csv --output outputs/predictions.csv

# 5. Diffusion analysis
python main.py diffusion data/synthetic_posts.csv

# 6. Start dashboard
python main.py dashboard

# 7. Start REST API
python main.py api
```

---

## Repository Structure

```
project/
├── config.yaml                      # Central configuration
├── main.py                          # CLI entry point
├── requirements.txt
├── README.md
├── data/                            # Datasets (CSV, JSON, JSONL)
├── models/                          # Saved model artifacts
├── outputs/                         # Reports, plots, predictions
├── notebooks/                       # Jupyter notebooks
├── tests/                           # Unit tests
├── app/
│   └── dashboard.py                 # Streamlit dashboard (9 tabs)
└── src/
    ├── data/
    │   ├── schema.py                 # Post schema, Label enum, DataFrame validator
    │   ├── loader.py                 # CSV / JSON / JSONL loaders
    │   ├── preprocessor.py           # Text normalization, deduplication, filtering
    │   └── synthetic_generator.py   # Demo dataset generator
    ├── features/
    │   ├── text_features.py          # Statistical + transformer embedding features
    │   ├── metadata_features.py      # Account, temporal, network behavior features
    │   └── feature_pipeline.py       # Unified pipeline with StandardScaler
    ├── detection/
    │   ├── models.py                 # LR, RF, XGB, LightGBM, ensemble factories
    │   ├── trainer.py                # Training pipeline, CV, metrics, saving
    │   ├── inference.py              # Single-post and batch prediction
    │   └── explainability.py         # SHAP-based feature attribution
    ├── diffusion/
    │   ├── graph_builder.py          # NetworkX interaction + cascade graph builders
    │   ├── metrics.py                # Depth, breadth, structural virality, centrality
    │   └── analysis.py               # AI vs human statistical comparison + reports
    ├── visualization/
    │   └── plots.py                  # Confusion matrix, ROC, boxplots, network graphs
    ├── api/
    │   └── server.py                 # FastAPI with /health /predict /predict_batch /analyze_diffusion /metrics
    └── utils/
        ├── config.py                 # YAML config loader
        └── logger.py                 # Loguru setup
```

---

## CLI Commands

### `generate` — Create synthetic dataset
```bash
python main.py generate \
  --n-posts 5000 \
  --ai-fraction 0.4 \
  --n-users 500 \
  --output data/my_dataset.csv
```

### `train` — Train detection models
```bash
# Basic training
python main.py train --data data/synthetic_posts.csv

# With transformer embeddings (GPU recommended)
python main.py train --data data/my_dataset.csv --use-embeddings

# Skip cross-validation for speed
python main.py train --data data/my_dataset.csv --no-cv
```

### `infer` — Batch inference
```bash
python main.py infer data/unseen_posts.csv \
  --model ensemble \
  --output outputs/predictions.csv
```

### `diffusion` — Network analysis
```bash
python main.py diffusion data/my_dataset.csv --output-dir outputs/
```

### `report` — Generate markdown report
```bash
python main.py report --output outputs/evaluation_report.md
```

### `api` — Start REST API
```bash
python main.py api --host 0.0.0.0 --port 8000
# Docs at http://localhost:8000/docs
```

### `dashboard` — Launch Streamlit UI
```bash
python main.py dashboard --port 8501
# Open http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single-post prediction (JSON body) |
| POST | `/predict_batch` | Batch prediction (CSV upload) |
| POST | `/analyze_diffusion` | Diffusion analysis (CSV upload) |
| GET | `/metrics` | Saved model metrics |

**Example: predict single post**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Leveraging synergies for optimal stakeholder outcomes.", "followers": 5000}'
```

---

## Data Schema

Each record should contain:

| Field | Type | Required |
|-------|------|----------|
| `post_id` | str | ✓ |
| `user_id` | str | ✓ |
| `text` | str | ✓ |
| `timestamp` | datetime | optional |
| `platform` | str | optional |
| `reply_to` | str | optional |
| `reshare_of` | str | optional |
| `engagement_count` | int | optional |
| `followers` | int | optional |
| `following` | int | optional |
| `label` | human/ai_generated/unknown | optional |

---

## Feature Families

### Text Features (17)
`char_count`, `token_count`, `unique_token_count`, `sentence_count`,
`avg_token_length`, `avg_sentence_length`, `sentence_length_variance`,
`type_token_ratio`, `repetition_rate`, `punct_ratio`,
`lexical_diversity_mattr`, `token_entropy`, `burstiness`,
`capital_ratio`, `url_count`, `mention_count`, `hashtag_count`

### Metadata Features (15)
`followers`, `following`, `engagement_count`, `ff_ratio`,
`engagement_rate`, `log_followers`, `log_following`, `log_engagement`,
`user_post_count`, `inter_post_time_mean`, `inter_post_time_var`,
`temporal_regularity`, `burst_score`, `hour_of_day`, `is_weekend`,
`reply_ratio`, `reshare_ratio`, `is_reply`, `is_reshare`

### Transformer Embeddings (optional)
768-dimensional vectors from `distilbert-base-uncased` (mean-pooled).

---

## Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline tabular |
| Random Forest | Ensemble tabular |
| XGBoost | Gradient boosting |
| Ensemble (Soft Voting) | LR + RF + XGB |

---

## Diffusion Metrics

- **Cascade depth** — longest reshare chain
- **Cascade breadth** — maximum nodes at any depth
- **Structural virality** — average pairwise distance (Wiener index)
- **Time to first spread** — hours to reach 10% of cascade
- **Time to peak spread** — hours to reach 90% of cascade
- **PageRank / betweenness centrality** — node influence
- **Community detection** — Louvain / greedy modularity

---

## Extending the Project

- **New model**: add a factory in `src/detection/models.py` and register in `DetectionTrainer.train_all()`
- **New features**: add extraction logic in `src/features/` and include in `FeaturePipeline._extract_all()`
- **Real dataset**: place any CSV/JSON/JSONL in `data/` and pass `--data` to `train`
- **Custom preprocessing**: extend `src/data/preprocessor.py`

---

## Dependencies

- Python 3.10+, PyTorch, Hugging Face Transformers, scikit-learn, XGBoost
- NetworkX, python-louvain, SHAP
- FastAPI + Uvicorn, Streamlit, Plotly, Matplotlib