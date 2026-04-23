# Cloud Deployment Architecture

## Reference Architecture (Cloud Agnostic)

1. Ingress/API Gateway routes external traffic to ALPR API service.
2. ALPR API pods run in Kubernetes (EKS/GKE/AKS) behind a load balancer.
3. Model artifacts are stored in object storage (S3/GCS/Blob).
4. Event stream and metadata are persisted in managed database (RDS/Cloud SQL/Azure DB).
5. Monitoring stack:
   - Prometheus for metrics scraping
   - Grafana for dashboards
   - Cloud-native logs in CloudWatch / Cloud Logging / Azure Monitor

## AWS Example

- Compute: EKS node group with GPU/CPU pools
- Registry: ECR
- Storage: S3 for checkpoints and datasets
- Database: RDS PostgreSQL (production replacement for SQLite)
- Monitoring: AMP + AMG or self-hosted Prometheus/Grafana

## GCP Example

- Compute: GKE Autopilot or Standard
- Registry: Artifact Registry
- Storage: GCS
- Database: Cloud SQL
- Monitoring: Cloud Monitoring + Managed Service for Prometheus

## Azure Example

- Compute: AKS
- Registry: Azure Container Registry (ACR)
- Storage: Azure Blob Storage
- Database: Azure Database for PostgreSQL
- Monitoring: Azure Monitor + Managed Prometheus + Grafana

## Deployment Steps

1. Build and push image through CI/CD.
2. Apply Kubernetes manifests in deployment/k8s.
3. Configure secrets (model path, DB creds, zone map).
4. Expose API and dashboard services.
5. Validate `/health`, `/predict/image`, `/metrics`.

## Production Hardening

- Replace SQLite with managed PostgreSQL.
- Enable TLS, auth, and role-based access for API.
- Add canary deployment for new model versions.
- Implement model registry and approval gates.
