# CV + CNN + Transformer-Based Number Plate Recognition System

An end-to-end Automatic Number Plate Recognition (ANPR/ALPR) system built for Indian plates with:

- Classical computer vision preprocessing (segmentation + corner detection)
- Hybrid deep learning architecture (CNN detection + Transformer sequence recognition)
- Real-time inference and multi-frame tracking
- Parking analytics (duplicates, occupancy, zone intelligence)
- REST API serving with FastAPI
- Dashboard with Streamlit
- Cloud-ready deployment via Docker, Kubernetes manifests, and GitHub Actions CI/CD

## 1. Repository Layout

```text
src/
  data/           # Splits, annotation validation, augmentation, CV localization
  models/         # Detector + Transformer recognizer + hybrid wrapper
  training/       # Losses, metrics, trainer
  inference/      # Tracking, post-processing, runtime pipeline
  analytics/      # Occupancy and event storage
  api/            # FastAPI service
  dashboard/      # Streamlit analytics dashboard
scripts/          # Data prep, train, evaluate, infer, drift monitor
configs/          # Training and deployment config
monitoring/       # Prometheus config and dashboards
deployment/k8s/   # Kubernetes manifests
.github/workflows # CI/CD pipelines
```

## 2. Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Data Preparation

Expected manifest CSV schema in `data/raw/manifest.csv`:

- `image_path`: absolute or project-relative path to image
- `plate_text`: ground-truth plate sequence
- `x_min`, `y_min`, `x_max`, `y_max`: plate bounding box

Run:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

This performs:
- annotation verification
- train/val/test split (70/15/15)
- normalized split manifests in `data/processed`

### Using dataclusterlabs Kaggle Dataset

If you are using https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset:

```bash
# 1) Download and extract
.venv/Scripts/kaggle.exe datasets download -d dataclusterlabs/indian-number-plates-dataset -p data/raw --unzip

# 2) Build manifest in project schema
python scripts/build_manifest_from_kaggle_datacluster.py --raw-dir data/raw --output data/raw/manifest.csv

# 3) Build clean + split CSVs
python scripts/prepare_data.py --config configs/default.yaml
```

The converter uses OCR XML annotations where number plate text is available, and writes rows with:
`image_path, plate_text, x_min, y_min, x_max, y_max`.

## 4. Train

```bash
python scripts/train.py --config configs/default.yaml
```

Outputs:
- best checkpoint: `checkpoints/best_model.pt`
- metrics: `artifacts/train_metrics.json`

## 5. Evaluate (includes real-world unseen images)

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
```

## 6. Real-Time Video Inference + Tracking

```bash
python scripts/infer_video.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --video path/to/video.mp4 --camera-id entry_cam_1
```

## 7. Run API + Dashboard

```bash
# API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Dashboard
streamlit run src/dashboard/app.py
```

## 8. Docker

```bash
docker build -t alpr-system:latest .
docker run -p 8000:8000 alpr-system:latest
```

## 9. Monitoring

- Prometheus endpoint: `/metrics` (FastAPI)
- Monitor API latency, requests, and custom recognition events.
- Track model drift and rolling accuracy by feeding labeled audit samples.

Drift monitor utility:

```bash
python scripts/monitor_drift.py --baseline 0.92 --current 0.86 --threshold 0.05
```

API drift endpoint:

`GET /analytics/drift?baseline_accuracy=0.92&current_accuracy=0.86&threshold=0.05`

## 10. CI/CD and Cloud

- CI: lint + tests + build validation via GitHub Actions.
- CD: Docker image push + Kubernetes deployment.
- Scheduled training pipeline: `.github/workflows/train-model.yml`.
- Cloud target examples: AWS EKS, GCP GKE, Azure AKS.
- Architecture reference: `docs/cloud_deployment_architecture.md`.

## 11. Branching Strategy

- `main`: production-ready
- `dev`: integration branch
- `feature/*`: isolated feature development

## 12. Notes

- The model is framework-complete and ready to train once dataset files are placed.
- For best results, include diverse parking lot lighting and angle conditions in `data/real_world_eval`.
