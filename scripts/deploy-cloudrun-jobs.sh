#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ghost-pitching-470315}"
REGION="${REGION:-us-east5}"
DATASET="${DATASET:-alpha_bsbl}"
VERSION="${VERSION:-v1}"
ANALYSIS_DATASET="${ANALYSIS_DATASET:-analysis_model}"
SRC_DATASET="${SRC_DATASET:-alpha_bsbl}"
PITCH_CORE_TABLE="${PITCH_CORE_TABLE:-pitch_core_v1}"
NF_TS_TABLE="${NF_TS_TABLE:-gold_newtforce_time_series}"

# All pitch types scored in accuracy mode
ACCURACY_PITCH_TYPES="Fastball,Sinker,Slider,Curveball,Changeup,Cutter,Splitter"

IMAGE="us-east5-docker.pkg.dev/ghost-pitching-470315/ghostlab/ghostlab-analytics-models:latest"

# Job name helpers
SLUG="${DATASET//_/-}"
COMMAND_JOB="accuracy-command-model-${SLUG}"
MISS_JOB="accuracy-miss-direction-${SLUG}"
BASELINE_ACC_JOB="nf-baseline-accuracy-${SLUG}"
SCORE_ACC_JOB="nf-score-accuracy-${SLUG}"
INTEL_JOB="coach-intel-build-${SLUG}"
BASELINE_VEL_JOB="nf-baseline-velocity-${SLUG}"
SCORE_VEL_JOB="nf-score-velocity-${SLUG}"
INTEL_VEL_JOB="nf-intel-velocity-${SLUG}"
NF_ANALYSIS_VEL_JOB="nf-nf-analysis-velocity-${SLUG}"

echo "================================================"
echo "GhostLab Analytics — Cloud Run Job Deploy"
echo "Project : ${PROJECT_ID}"
echo "Region  : ${REGION}"
echo "Dataset : ${DATASET}  Version: ${VERSION}"
echo "Image   : ${IMAGE}"
echo "================================================"

gcloud config set project "${PROJECT_ID}"

# ── Build & push image ───────────────────────────────────────────────────────
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}"

# ── Shared env vars ──────────────────────────────────────────────────────────
BASE_ENV="GCP_PROJECT=${PROJECT_ID},BQ_LOCATION=${REGION},BQ_ANALYSIS_DATASET=${ANALYSIS_DATASET},BQ_SRC_DATASET=${SRC_DATASET}"

# ── 1. Command Model ─────────────────────────────────────────────────────────
echo "Deploying: ${COMMAND_JOB}"
gcloud run jobs deploy "${COMMAND_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "4Gi" --cpu "2" \
  --max-retries "0" --task-timeout "3600s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "command-model,--dataset,${DATASET},--version,${VERSION},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE}"

# ── 2. Miss Direction Model ──────────────────────────────────────────────────
echo "Deploying: ${MISS_JOB}"
gcloud run jobs deploy "${MISS_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "4Gi" --cpu "2" \
  --max-retries "0" --task-timeout "3600s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "miss-direction,--dataset,${DATASET},--version,${VERSION},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE}"

# ── 3. NF Accuracy Baseline ──────────────────────────────────────────────────
# NOTE: runs parallel with jobs 1+2 (no shared dependencies)
echo "Deploying: ${BASELINE_ACC_JOB}"
gcloud run jobs deploy "${BASELINE_ACC_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "8Gi" --cpu "4" \
  --max-retries "0" --task-timeout "7200s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "baseline,--dataset,${DATASET},--version,${VERSION},--mode,accuracy,--pitch-types,${ACCURACY_PITCH_TYPES},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE},--nf-time-series-table,${NF_TS_TABLE}"

# ── 4. NF Accuracy Scores ────────────────────────────────────────────────────
# NOTE: depends on job 3 (baseline must finish first)
echo "Deploying: ${SCORE_ACC_JOB}"
gcloud run jobs deploy "${SCORE_ACC_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "8Gi" --cpu "4" \
  --max-retries "0" --task-timeout "7200s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "score-nf,--dataset,${DATASET},--version,${VERSION},--mode,accuracy,--pitch-types,${ACCURACY_PITCH_TYPES},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE},--nf-time-series-table,${NF_TS_TABLE}"

# ── 5. Coach Intelligence Build ──────────────────────────────────────────────
# NOTE: depends on jobs 1+2+4 (all models + scores must finish first)
echo "Deploying: ${INTEL_JOB}"
gcloud run jobs deploy "${INTEL_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "2Gi" --cpu "1" \
  --max-retries "0" --task-timeout "1800s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "intel-build,--dataset,${DATASET},--version,${VERSION},--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET}"

# ── 6. NF Velocity Baseline ──────────────────────────────────────────────────
# NOTE: independent of accuracy pipeline — runs parallel with everything above
echo "Deploying: ${BASELINE_VEL_JOB}"
gcloud run jobs deploy "${BASELINE_VEL_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "8Gi" --cpu "4" \
  --max-retries "0" --task-timeout "7200s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "velocity-baseline,--dataset,${DATASET},--version,${VERSION},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE},--nf-time-series-table,${NF_TS_TABLE}"

# ── 7. NF Velocity Scores ────────────────────────────────────────────────────
# NOTE: depends on job 6 (velocity baseline must finish first)
echo "Deploying: ${SCORE_VEL_JOB}"
gcloud run jobs deploy "${SCORE_VEL_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "8Gi" --cpu "4" \
  --max-retries "0" --task-timeout "7200s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "velocity-score,--dataset,${DATASET},--version,${VERSION},--write-bq,--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--pitch-core-table,${PITCH_CORE_TABLE},--nf-time-series-table,${NF_TS_TABLE}"

# ── 8. Velocity Intelligence Build ───────────────────────────────────────────
# NOTE: depends on job 6+7 (velocity baseline + scores must finish first)
echo "Deploying: ${INTEL_VEL_JOB}"
gcloud run jobs deploy "${INTEL_VEL_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "2Gi" --cpu "1" \
  --max-retries "0" --task-timeout "1800s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "velocity-intel,--dataset,${DATASET},--version,${VERSION},--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET}"

# ── 9. Velocity NF Correlation Analysis ──────────────────────────────────────
# NOTE: depends on job 7 (scores_model_velocity must exist — has nf_file_name + pitch_velocity)
echo "Deploying: ${NF_ANALYSIS_VEL_JOB}"
gcloud run jobs deploy "${NF_ANALYSIS_VEL_JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory "8Gi" --cpu "4" \
  --max-retries "0" --task-timeout "3600s" \
  --set-env-vars "${BASE_ENV}" \
  --command "ghostlab" \
  --args "velocity-nf-analysis,--dataset,${DATASET},--version,${VERSION},--gcp-project,${PROJECT_ID},--bq-location,${REGION},--bq-analysis-dataset,${ANALYSIS_DATASET},--bq-src-dataset,${SRC_DATASET}"

echo "================================================"
echo "All jobs deployed. Nightly execution order:"
echo "  Step A (parallel): ${COMMAND_JOB}"
echo "                     ${MISS_JOB}"
echo "                     ${BASELINE_ACC_JOB}"
echo "  Step B:            ${SCORE_ACC_JOB}   (after accuracy baseline)"
echo "  Step C:            ${INTEL_JOB}        (after command + miss + accuracy scores)"
echo "  Step D:            ${BASELINE_VEL_JOB} (after pitch_core)"
echo "  Step E:            ${SCORE_VEL_JOB}    (after velocity baseline)"
echo "  Step F:            ${INTEL_VEL_JOB}    (after velocity scores)"
echo "  Step G:            ${NF_ANALYSIS_VEL_JOB} (after velocity scores)"
echo "================================================"
