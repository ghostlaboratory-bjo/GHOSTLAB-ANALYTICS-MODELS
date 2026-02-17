from __future__ import annotations

from dataclasses import dataclass
from google.cloud import bigquery

@dataclass(frozen=True)
class BQConfig:
    project: str
    location: str | None = None

def bq_client(cfg: BQConfig) -> bigquery.Client:
    # Location matters for multi-region datasets (yours is us-east5)
    return bigquery.Client(project=cfg.project, location=cfg.location)
