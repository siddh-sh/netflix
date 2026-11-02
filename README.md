# Netflix Data Analysis — MLOps Project

End-to-end MLOps example on the Netflix titles dataset (`data/netflix_titles.csv`):
exploratory analytics, a Movie-vs-TV-Show text classifier, a Streamlit app, and the
infrastructure around it.

| Concern | Tooling |
|---|---|
| Version control | Git / GitHub |
| CI/CD | GitHub Actions (`.github/workflows`) |
| Packaging | Docker (model is trained at image build time) |
| Orchestration | Kubernetes manifests + kustomize (`k8s/`) |
| Infrastructure | Terraform on AWS: VPC, EKS, ECR (`terraform/`) |
| Monitoring | Prometheus + Grafana (`monitoring/`) |
| App | Streamlit (`app/streamlit_app.py`) |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python -m src.train                      # writes models/model.joblib + metrics.json
streamlit run app/streamlit_app.py       # http://localhost:8501
```

## Run with monitoring (Docker)

```bash
docker compose up --build
```

- App: http://localhost:8501 (metrics at :8000/metrics)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin) — "Netflix MLOps" dashboard is pre-provisioned

Metrics exported: page views, prediction counts by label, prediction latency,
prediction confidence, model accuracy/F1, dataset size.

## Kubernetes

```bash
kubectl kustomize --load-restrictor LoadRestrictionsNone k8s | kubectl apply -f -
kubectl -n netflix get svc          # LoadBalancer addresses for app and Grafana
```

Change the Grafana password in `k8s/kustomization.yaml` before using this anywhere real.

## Terraform

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
terraform init && terraform apply
$(terraform output -raw kubeconfig_command)
```

This creates real, billable AWS resources (EKS, NAT gateway). Run `terraform destroy` when done.

## CI/CD

- `ci.yml`: ruff + pytest, terraform fmt/validate, kustomize build, Docker build.
- `cd.yml`: on `main`, pushes the image to `ghcr.io/<owner>/netflix` (`latest` and commit SHA).
  Deployment to a cluster is opt-in: set repo variable `DEPLOY_ENABLED=true` and secret
  `KUBE_CONFIG` (base64 kubeconfig).
