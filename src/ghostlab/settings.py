# src/ghostlab/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env automatically (local-first workflow)
load_dotenv()

def _csv(v: str | None) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]

@dataclass(frozen=True)
class Settings:
    # repo paths
    repo_root: Path = Path(__file__).resolve().parents[2]  # repo root (…/src/ghostlab/settings.py -> …/)
    data_processed_root: Path = Path(os.getenv("LOCAL_DATA_DIR", "./data")).resolve()
    outputs_dir: Path = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()
    artifacts_dir: Path = Path(os.getenv("ARTIFACT_DIR", "./artifacts")).resolve()

    # gcp/bq
    gcp_project: str = os.getenv("GCP_PROJECT", "").strip()
    bq_location: str = os.getenv("BQ_LOCATION", "us-east5").strip()
    bq_src_dataset: str = os.getenv("BQ_SRC_DATASET", "").strip()
    bq_analysis_dataset: str = os.getenv("BQ_ANALYSIS_DATASET", "analysis_model").strip()
    bq_pitch_core_table: str = os.getenv("BQ_PITCH_CORE_TABLE", "pitch_core_v1").strip()
    bq_nf_time_series_table: str = os.getenv("BQ_NF_TIME_SERIES_TABLE", "newtforce_time_series").strip()

    # baseline build defaults
    target_timesteps: int = int(os.getenv("TARGET_TIMESTEPS", "700"))
    baseline_mode: str = os.getenv("BASELINE_MODE", "velocity").strip().lower()
    baseline_max_pitches_per_group: int = int(os.getenv("BASELINE_MAX_PITCHES_PER_GROUP", "50"))
    baseline_pitch_types: list[str] = tuple(_csv(os.getenv("BASELINE_PITCH_TYPES", "Fastball,Sinker")))

# ✅ This is what your CLI expects to import
settings = Settings()
