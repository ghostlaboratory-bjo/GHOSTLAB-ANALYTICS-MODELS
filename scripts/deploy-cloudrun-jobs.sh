#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ghost-pitching-470315}"
REGION="${REGION:-us-east5}"
DATASET="${DATASET:-alpha_bsbl}"
VERSION="${VERSION:-v1}"
ANALYSIS_DATASET="${ANALYSIS_DATASET:-analysis_model}"
PITCH_CORE_TABLE="${PITCH_CORE_TABLE:-pitch_core_v1}"

IMAGE="us-east5-docker.pkg.dev/ghost-pitching-470315/ghostlab/ghostlab-analytics-models:latest"

COMMAND_JOB="accuracy-command-model-${DATASET//_/-}"
MISS_JOB="accuracy-miss-direction-${DATASET//_/-}"

echo "Using project: ${PROJECT_ID}"
echo "Using region: ${REGION}"
echo "Building image: ${IMAGE}"

gcloud config set project "${PROJECT_ID}"

gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}"

echo "Deploying Cloud Run Job: ${COMMAND_JOB}"

gcloud run jobs deploy "${COMMAND_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "4Gi" \
  --cpu "2" \
  --max-retries "0" \
  --task-timeout "3600s" \
  --set-env-vars "GCP_PROJECT=${PROJECT_ID},BQ_LOCATION=${REGION},BQ_ANALYSIS_DATASET=${ANALYSIS_DATASET}" \
  --command "ghostlab" \
  --args "command-model,--dataset,${DATASET},--version,${VERSION},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE}"

echo "Deploying Cloud Run Job: ${MISS_JOB}"

gcloud run jobs deploy "${MISS_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "4Gi" \
  --cpu "2" \
  --max-retries "0" \
  --task-timeout "3600s" \
  --set-env-vars "GCP_PROJECT=${PROJECT_ID},BQ_LOCATION=${REGION},BQ_ANALYSIS_DATASET=${ANALYSIS_DATASET}" \
  --command "ghostlab" \
  --args "miss-direction,--dataset,${DATASET},--version,${VERSION},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE}"

echo "Done."
echo "Command model job: ${COMMAND_JOB}"
echo "Miss direction job: ${MISS_JOB}"