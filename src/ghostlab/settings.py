from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env if present


def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # backend mode
    data_backend: str = (_env("DATA_BACKEND", "local") or "local").lower()

    # directories (relative to repo root by default)
    local_data_dir: Path = Path(_env("LOCAL_DATA_DIR", "./data") or "./data")
    output_dir: Path = Path(_env("OUTPUT_DIR", "./outputs") or "./outputs")
    artifact_dir: Path = Path(_env("ARTIFACT_DIR", "./artifacts") or "./artifacts")
    log_dir: Path = Path(_env("LOG_DIR", "./logs") or "./logs")

    # optional GCP fields (only used later)
    gcp_project: str | None = _env("GCP_PROJECT")
    bq_location: str | None = _env("BQ_LOCATION")
    bq_dataset: str | None = _env("BQ_DATASET")


def get_settings() -> Settings:
    return Settings()
