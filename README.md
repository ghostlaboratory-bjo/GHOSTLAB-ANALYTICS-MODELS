# GhostLab Analytics Models

Local-first analytics and ML pipelines for Ghost Laboratory.

This repo builds and runs the model jobs that feed BigQuery tables used by the GhostLab website, including command/accuracy models, miss direction models, NewtForce best-pattern baselines, and pitch deviation scoring.

---

## Repo Purpose

This repo is responsible for model creation and scoring jobs.

Current production jobs include:

- Command / accuracy model
- Miss direction model
- NewtForce best pattern baseline generation
- NewtForce pitch deviation scoring

These jobs write output tables into BigQuery, primarily under:

```txt
ghost-pitching-470315.analysis_model




GCP Project: ghost-pitching-470315
Region: us-east5
BigQuery Location: us-east5
Team Dataset: alpha_bsbl
Analysis Dataset: analysis_model
Feature Version: v1
Pitch Core Table: pitch_core_v1



Local Setup in VS Code

Open this repo in VS Code:

cd ~/Projects/GHOSTLAB-ANALYTICS-MODELS
code .

Create / activate the virtual environment:

source .venv/bin/activate

Install the package locally:

python -m pip install --upgrade pip
python -m pip install -e .

Verify the CLI:

ghostlab doctor

Expected output:

OK: ghostlab CLI is wired up.
Google Cloud Local Auth

Local model runs use your Google Application Default Credentials.

If BigQuery auth fails, run:

gcloud auth application-default login
gcloud config set project ghost-pitching-470315

Confirm project:

gcloud config get-value project

Expected:

ghost-pitching-470315
Local Manual Model Runs
Command / Accuracy Model
ghostlab command-model \
  --dataset alpha_bsbl \
  --version v1 \
  --write-bq \
  --gcp-project ghost-pitching-470315 \
  --bq-location us-east5 \
  --bq-analysis-dataset analysis_model \
  --pitch-core-table pitch_core_v1

Output table:

ghost-pitching-470315.analysis_model.model_command_scores_v1
Miss Direction Model
ghostlab miss-direction \
  --dataset alpha_bsbl \
  --version v1 \
  --write-bq \
  --gcp-project ghost-pitching-470315 \
  --bq-location us-east5 \
  --bq-analysis-dataset analysis_model \
  --pitch-core-table pitch_core_v1

Output table:

ghost-pitching-470315.analysis_model.model_miss_direction_v1
Cloud Run Jobs

The production model jobs run as Google Cloud Run Jobs.

Current deployed jobs:

accuracy-command-model-alpha-bsbl
accuracy-miss-direction-alpha-bsbl

List jobs:

gcloud run jobs list \
  --region us-east5 \
  --project ghost-pitching-470315
Manual Cloud Run Job Execution
Run Command Model Job
gcloud run jobs execute accuracy-command-model-alpha-bsbl \
  --region us-east5 \
  --project ghost-pitching-470315 \
  --wait
Run Miss Direction Job
gcloud run jobs execute accuracy-miss-direction-alpha-bsbl \
  --region us-east5 \
  --project ghost-pitching-470315 \
  --wait

Describe a specific execution:

gcloud run jobs executions describe EXECUTION_NAME \
  --region us-east5 \
  --project ghost-pitching-470315
Docker / Cloud Run Deployment

This repo uses Artifact Registry, not legacy gcr.io.

Production image:

us-east5-docker.pkg.dev/ghost-pitching-470315/ghostlab/ghostlab-analytics-models:latest

Artifact Registry repo:

ghostlab

Create the repo once if needed:

gcloud artifacts repositories create ghostlab \
  --repository-format=docker \
  --location=us-east5 \
  --description="GhostLab ML Jobs"

Deploy / update Cloud Run Jobs:

chmod +x scripts/deploy-cloudrun-jobs.sh
./scripts/deploy-cloudrun-jobs.sh

The deploy script builds the Docker image, pushes it to Artifact Registry, and deploys:

accuracy-command-model-alpha-bsbl
accuracy-miss-direction-alpha-bsbl
Required Files for Deployment
Dockerfile
.dockerignore
scripts/deploy-cloudrun-jobs.sh
pyproject.toml
src/ghostlab/cli.py
src/ghostlab/settings.py
Nightly Workflow Integration

The accuracy jobs are called from:

alpha_baseball_etl_workflow.yaml

Current workflow sequence:

FTP ingestion
Google Drive ingestion
GCS to BigQuery ingestion
Bronze Trackman load
Silver Trackman build
Silver NewtForce build
Gold Trackman pitch paths
Gold pitching summary
Gold NewtForce time series
Command model
Miss direction model

Deploy workflow:

gcloud workflows deploy alpha-baseball-etl-workflow \
  --source=alpha_baseball_etl_workflow.yaml \
  --location=us-east5 \
  --project=ghost-pitching-470315

Run workflow manually:

gcloud workflows run alpha-baseball-etl-workflow \
  --location=us-east5 \
  --project=ghost-pitching-470315 \
  --data='{
    "projectId":"ghost-pitching-470315",
    "region":"us-east5",
    "dataset":"alpha_bsbl",
    "bqLocation":"us-east5",
    "pollIntervalMinutes":5
  }'
Current Accuracy Model Outputs
Command Model

Table:

analysis_model.model_command_scores_v1

Purpose:

Predicts command / miss distance and produces command scoring outputs.

Validated local run:

Rows modeled: 21,264
MAE: ~3.43 inches
R2: ~0.734

Important features from validation:

horz_rel_angle_delta
vert_rel_angle_delta
HorzApprAngle
VertApprAngle
RelSide
VertRelAngle
rel_side_delta
Miss Direction Model

Table:

analysis_model.model_miss_direction_v1

Purpose:

Classifies predicted horizontal and vertical miss direction.

Horizontal classes:

Arm-Side
Glove-Side
In-Zone

Vertical classes:

High
Low
In-Zone

Validated local run:

Horizontal accuracy: ~0.861
Vertical accuracy: ~0.880

Important features from validation:

horz_rel_angle_delta
vert_rel_angle_delta
HorzApprAngle
VertApprAngle
HorzRelAngle
VertRelAngle
RelSide
RelHeight
Common Issues
ghostlab: command not found

Activate the virtual environment and reinstall:

source .venv/bin/activate
python -m pip install -e .

Then:

ghostlab doctor
BigQuery Reauthentication Error

Error:

Reauthentication is needed. Please run gcloud auth application-default login

Fix:

gcloud auth application-default login
gcloud config set project ghost-pitching-470315
Cloud Run Job Not Found

Error:

Resource accuracy-command-model-alpha-bsbl of kind JOB does not exist

Fix:

./scripts/deploy-cloudrun-jobs.sh

Then verify:

gcloud run jobs list \
  --region us-east5 \
  --project ghost-pitching-470315
Image Push Fails to gcr.io

Use Artifact Registry instead:

us-east5-docker.pkg.dev/ghost-pitching-470315/ghostlab/ghostlab-analytics-models:latest

Do not use:

gcr.io/ghost-pitching-470315/ghostlab-analytics-models:latest
Git Workflow

Check current branch/status:

git status
git branch

Commit changes:

git add .
git commit -m "Update analytics model deployment"
git push origin main

If push is rejected because remote has newer commits:

git pull --rebase origin main
git push origin main

If generated files block rebase, move them temporarily:

mv env.cloudrun.yaml env.cloudrun.yaml.local-backup
git pull --rebase origin main
git push origin main
mv env.cloudrun.yaml.local-backup env.cloudrun.yaml
Next Planned Work

Next step is to create a BigQuery stored procedure that builds a final website-facing table:

analysis_model.model_pitch_intelligence_v1

This table should combine:

gold_pitching_summary
model_command_scores_v1
model_miss_direction_v1
NewtForce best pattern / deviation outputs

Goal:

Move query and insight logic out of the website UI/API layer and into BigQuery so the website can read a clean precomputed table.

Target website behavior:

Coach Intelligence page reads one clean model table
Session filter works directly on session_date
Command score, miss direction, mechanics driver, and teaching priority are precomputed
Recommended Operating Flow

For local development:

source .venv/bin/activate
python -m pip install -e .
ghostlab doctor
ghostlab command-model ...
ghostlab miss-direction ...

For production deployment:

./scripts/deploy-cloudrun-jobs.sh

For manual production execution:

gcloud run jobs execute accuracy-command-model-alpha-bsbl --region us-east5 --project ghost-pitching-470315 --wait
gcloud run jobs execute accuracy-miss-direction-alpha-bsbl --region us-east5 --project ghost-pitching-470315 --wait

For nightly automation:

Google Scheduler → Google Workflow → Cloud Run Jobs + BigQuery Stored Procedures


