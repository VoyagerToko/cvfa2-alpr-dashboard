# System Architecture

## Overview

The system is a hybrid ALPR stack with four tiers:

1. Data and preprocessing
2. Hybrid model (CNN detector + Transformer recognizer)
3. Real-time inference and analytics
4. Cloud-native serving and monitoring

## Data Flow

1. Raw images and annotations are ingested from `data/raw/manifest.csv`.
2. Annotation verifier filters broken rows and normalizes plate text.
3. Split generator creates train/val/test with 70/15/15.
4. Runtime CV localization performs:
   - Threshold segmentation
   - Region growing
   - Edge-based segmentation (Canny + contours)
   - Harris corner extraction
   - Corner-assisted boundary approximation and optional perspective correction
5. Hybrid model predicts:
   - Plate box localization
   - Character sequence logits for CTC decoding
6. Post-processing layer handles duplicates and tracking.
7. Analytics stores events and computes occupancy by zone.

## Model Blocks

- Detector: CNN global features -> normalized bbox + confidence
- Recognizer: CNN feature map -> sequence projection -> Transformer encoder -> character logits
- Decoder: CTC greedy decoding

## Runtime Components

- FastAPI (`/predict/image`, `/events`, `/analytics/occupancy`)
- Streamlit dashboard for monitoring occupancy and plate history
- SQLite for lightweight event persistence
- Prometheus metrics endpoint via FastAPI instrumentation

## Deployment

- Docker image with API and dashboard entrypoints
- Kubernetes manifests for API, dashboard, and HPA
- GitHub Actions CI/CD for lint, tests, image build, and deployment
