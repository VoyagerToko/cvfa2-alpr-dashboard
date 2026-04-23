# MLOps and Operations Guide

## Version Control Strategy

- `main`: production branch
- `dev`: integration branch
- `feature/*`: development branches

Merge policy:
- Feature -> dev via PR with CI checks
- Dev -> main after regression testing and approval

## CI Pipeline

`ci.yml` stages:
1. Install dependencies
2. Lint (`ruff`)
3. Unit tests (`pytest`)
4. Build sanity (`compileall`)

## CD Pipeline

`cd.yml` stages:
1. Build Docker image
2. Push image to GHCR
3. Deploy Kubernetes manifests (if `KUBE_CONFIG_DATA` secret is set)

## Training Automation

Recommended schedule:
- Nightly training on latest data
- Weekly retraining with drift review

Use script sequence:
1. `python scripts/prepare_data.py`
2. `python scripts/train.py`
3. `python scripts/evaluate.py`

## Monitoring

Track:
- API latency and request throughput via `/metrics`
- Character accuracy and full-plate accuracy from evaluation logs
- Drift indicators using rolling validation and real-world audit samples

## Model Drift Strategy

- Maintain a monthly labeled unseen set from parking lot cameras.
- Compare current metrics vs baseline model.
- Trigger retraining when plate accuracy drops more than 5%.

## Incident Response

1. If API errors spike:
   - Check deployment rollout status
   - Check model checkpoint availability
2. If accuracy degrades:
   - Run `scripts/evaluate.py` on audit set
   - Verify camera quality and localization quality
3. If duplicate rate spikes:
   - Tune duplicate window and Levenshtein threshold in config
