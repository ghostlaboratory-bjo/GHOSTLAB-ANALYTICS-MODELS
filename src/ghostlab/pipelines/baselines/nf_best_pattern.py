# src/ghostlab/pipelines/baselines/nf_best_pattern.py
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.interpolate import interp1d
from rich.console import Console

from ghostlab.io.bq import BQConfig, bq_client

console = Console()

TS_FEATURES = ["fx_lb", "fy_lb", "fz_lb", "x_in", "y_in"]
FORCE_FEATURES = {"fx_lb", "fy_lb", "fz_lb"}


@dataclass(frozen=True)
class BaselineBuildConfig:
    # Identity
    dataset_id: str
    feature_version: str

    # Resampling
    target_timesteps: int = 700

    # Mode: velocity|accuracy
    mode: str = "velocity"

    # Inputs
    # ✅ Default Fastball only (unless CLI passes --pitch-types)
    pitch_types: Tuple[str, ...] = ("Fastball",)
    max_pitches_per_group: int = 50

    # BQ connection + sources
    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_src_dataset: str = ""  # where NF TS lives
    bq_analysis_dataset: str = "analysis_model"
    pitch_core_table: str = "pitch_core_v1"
    nf_time_series_table: str = "gold_newtforce_time_series"

    # BQ outputs (base model tables)
    write_bq: bool = False
    bq_out_dataset: str = "analysis_model"
    bq_write_disposition: str = "WRITE_TRUNCATE"  # full rebuild default
    bq_baseline_table_base: str = "baseline_model"  # -> baseline_model_velocity|accuracy

    # Local output
    local_root: Path = Path("./data/processed")

    # Performance
    write_batch_rows: int = 200_000  # TS rows per load job


def _parse_csv_list(v: str | None) -> Tuple[str, ...]:
    if not v:
        return tuple()
    return tuple(x.strip() for x in v.split(",") if x.strip())


def _baseline_kind(cfg: BaselineBuildConfig) -> str:
    return "best_accuracy_pct" if (cfg.mode or "").strip().lower() == "accuracy" else "best_velocity_pct"


def _dest_dir(cfg: BaselineBuildConfig) -> Path:
    return cfg.local_root / cfg.dataset_id / "baselines" / cfg.feature_version


def _pitch_core_fq(cfg: BaselineBuildConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{cfg.pitch_core_table}"


def _ts_fq(cfg: BaselineBuildConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_src_dataset}.{cfg.nf_time_series_table}"


def _bq_out_fq(cfg: BaselineBuildConfig, table: str) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_out_dataset}.{table}"


def _baseline_table_name(cfg: BaselineBuildConfig) -> str:
    mode = (cfg.mode or "").strip().lower()
    return f"{cfg.bq_baseline_table_base}_{mode}"  # baseline_model_velocity|accuracy


def _make_baseline_id(cfg: BaselineBuildConfig, player_full_name: str, pitch_type: str) -> str:
    # Stable id so scoring can join baseline_id from meta map
    return f"{cfg.dataset_id}|{cfg.feature_version}|{_baseline_kind(cfg)}|{player_full_name}|{pitch_type}"


def _resample_one_pitch(pitch_ts: pd.DataFrame, weight_lb: float, T: int) -> np.ndarray | None:
    if pitch_ts is None or pitch_ts.empty or len(pitch_ts) < 2:
        return None
    if not weight_lb or weight_lb <= 0:
        return None

    pitch_ts = pitch_ts.sort_values("time_s").drop_duplicates(subset=["time_s"])
    if len(pitch_ts) < 2:
        return None

    t = pitch_ts["time_s"].to_numpy(dtype=np.float64)
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or (tmax - tmin) <= 1e-9:
        return None

    new_t = np.linspace(tmin, tmax, T)
    out = np.zeros((len(TS_FEATURES), T), dtype=np.float32)

    for j, feat in enumerate(TS_FEATURES):
        v = pitch_ts[feat].to_numpy(dtype=np.float64)
        if np.all(~np.isfinite(v)):
            return None

        f = interp1d(
            t,
            v,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
            assume_sorted=True,
        )
        r = f(new_t).astype(np.float32)

        if feat in FORCE_FEATURES:
            out[j, :] = r / float(weight_lb)
        else:
            out[j, :] = r

    if not np.isfinite(out).all():
        return None
    return out


def _fetch_baseline_candidates(cfg: BaselineBuildConfig) -> pd.DataFrame:
    """
    Picks candidate pitches using pitch_core flags:
      velocity: is_velocity_baseline_candidate = 1
      accuracy: is_accuracy_baseline_candidate = 1

    ✅ Hard-gates to Trackman + NF TS + weight (defensive).
    """
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    mode = (cfg.mode or "").strip().lower()
    flag_col = "is_accuracy_baseline_candidate" if mode == "accuracy" else "is_velocity_baseline_candidate"

    sql = f"""
    SELECT
      dataset_id,
      player_full_name,
      TaggedPitchType AS pitch_type,
      PitchID,
      nf_file_name,
      player_weight_lb,
      pitch_velocity,
      miss_distance_in,
      {flag_col} AS is_baseline_candidate
    FROM `{_pitch_core_fq(cfg)}`
    WHERE dataset_id = @dataset_id
      AND has_trackman = TRUE
      AND has_nf_timeseries = TRUE
      AND nf_file_name IS NOT NULL
      AND player_weight_lb IS NOT NULL
      AND player_weight_lb > 0
      AND TaggedPitchType IN UNNEST(@pitch_types)
      AND {flag_col} = 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", cfg.dataset_id),
            bigquery.ArrayQueryParameter("pitch_types", "STRING", list(cfg.pitch_types)),
        ]
    )

    df = client.query(sql, job_config=job_config).to_dataframe()

    if not df.empty:
        df["pitch_velocity"] = pd.to_numeric(df["pitch_velocity"], errors="coerce")
        df["miss_distance_in"] = pd.to_numeric(df["miss_distance_in"], errors="coerce")
        df["player_weight_lb"] = pd.to_numeric(df["player_weight_lb"], errors="coerce")
        if mode == "accuracy":
            df = df[np.isfinite(df["miss_distance_in"].to_numpy(dtype=np.float64))].copy()

    return df


def _fetch_time_series(cfg: BaselineBuildConfig, file_names: List[str]) -> pd.DataFrame:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    sql = f"""
    SELECT
      file_name,
      time_s,
      fx_lb,
      fy_lb,
      fz_lb,
      x_in,
      y_in
    FROM `{_ts_fq(cfg)}`
    WHERE file_name IN UNNEST(@file_names)
    ORDER BY file_name, time_s
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("file_names", "STRING", file_names)]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def _write_df_to_bq(cfg: BaselineBuildConfig, df: pd.DataFrame, table_fq: str, disposition: str) -> None:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    job_config = bigquery.LoadJobConfig(write_disposition=disposition)
    job = client.load_table_from_dataframe(df, table_fq, job_config=job_config)
    job.result()


def build_nf_best_pattern_baselines(cfg: BaselineBuildConfig) -> Path:
    out_dir = _dest_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    baseline_kind = _baseline_kind(cfg)
    mode = (cfg.mode or "").strip().lower()

    console.print(f"[bold]Baseline build[/bold] dataset={cfg.dataset_id} version={cfg.feature_version} mode={mode}")
    console.print(f"Run: {run_id}")

    candidates = _fetch_baseline_candidates(cfg)
    if candidates.empty:
        raise RuntimeError("No baseline candidates found. Check pitch_core flags and pitch types.")

    # Sort so we pick best candidates first
    if mode == "accuracy":
        candidates = candidates.sort_values(
            ["player_full_name", "pitch_type", "miss_distance_in", "pitch_velocity"],
            ascending=[True, True, True, False],
        )
    else:
        candidates = candidates.sort_values(
            ["player_full_name", "pitch_type", "pitch_velocity", "miss_distance_in"],
            ascending=[True, True, False, True],
        )

    candidates = (
        candidates.groupby(["player_full_name", "pitch_type"], as_index=False)
        .head(cfg.max_pitches_per_group)
        .reset_index(drop=True)
    )

    file_names = candidates["nf_file_name"].dropna().unique().tolist()
    console.print(f"Candidates: {len(candidates):,} rows | unique files: {len(file_names):,}")

    ts = _fetch_time_series(cfg, file_names)
    console.print(f"Fetched time series rows: {len(ts):,}")

    T = int(cfg.target_timesteps)
    time_s_norm = np.linspace(0.0, 1.0, T).astype(np.float32)
    ts_by_file = {k: g for k, g in ts.groupby("file_name")}

    rows_ts: List[dict] = []
    skipped_empty = 0
    skipped_bad = 0

    # Build mean waveform per (player,pitch_type)
    for (player, pitch_type), g in candidates.groupby(["player_full_name", "pitch_type"]):
        baseline_id = _make_baseline_id(cfg, player, pitch_type)
        pitch_arrays = []
        used_pitch_ids: List[str] = []

        for r in g.itertuples(index=False):
            pid = str(r.PitchID)
            fn = str(r.nf_file_name)
            w = float(r.player_weight_lb) if pd.notna(r.player_weight_lb) else 0.0

            pitch_ts = ts_by_file.get(fn)
            if pitch_ts is None or pitch_ts.empty:
                skipped_empty += 1
                continue

            arr = _resample_one_pitch(pitch_ts, w, T)
            if arr is None:
                skipped_bad += 1
                continue

            pitch_arrays.append(arr)
            used_pitch_ids.append(pid)

        if not pitch_arrays:
            continue

        stack = np.stack(pitch_arrays, axis=0)  # (n,c,t)
        mean_wave = stack.mean(axis=0)  # (c,t)
        n_used = int(stack.shape[0])

        for t in range(T):
            rows_ts.append(
                {
                    "baseline_id": baseline_id,
                    "dataset_id": cfg.dataset_id,
                    "feature_version": cfg.feature_version,
                    "mode": mode,
                    "baseline_kind": baseline_kind,
                    "player_full_name": player,
                    "pitch_type": pitch_type,
                    "timestep": int(t),
                    "time_s": float(time_s_norm[t]),
                    "fx_lb": float(mean_wave[0, t]),
                    "fy_lb": float(mean_wave[1, t]),
                    "fz_lb": float(mean_wave[2, t]),
                    "x_in": float(mean_wave[3, t]),
                    "y_in": float(mean_wave[4, t]),
                    "n_pitches_used": n_used,
                    "created_at": created_at,
                    "run_id": run_id,
                }
            )

    df_ts = pd.DataFrame(rows_ts)
    if df_ts.empty:
        raise RuntimeError("No baselines produced. Likely missing time series for candidate files.")

    # Local outputs (debuggable)
    out_path = out_dir / "nf_best_pattern_baselines.parquet"
    df_ts.to_parquet(out_path, index=False)

    manifest = {
        "dataset_id": cfg.dataset_id,
        "feature_version": cfg.feature_version,
        "mode": mode,
        "baseline_kind": baseline_kind,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "target_timesteps": T,
        "channels": TS_FEATURES,
        "player_pitchtype_groups": int(df_ts[["player_full_name", "pitch_type"]].drop_duplicates().shape[0]),
        "players": int(df_ts["player_full_name"].nunique()),
        "skipped_empty_ts": skipped_empty,
        "skipped_bad_resample": skipped_bad,
        "output_file": str(out_path),
    }
    with open(out_dir / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[green]✅ Baselines written:[/green] {out_path}")
    console.print(f"Groups: {manifest['player_pitchtype_groups']} | Players: {manifest['players']}")

    # BigQuery write -> base model table for this mode
    if cfg.write_bq:
        table_name = _baseline_table_name(cfg)
        table_fq = _bq_out_fq(cfg, table_name)

        console.print(f"Writing to BigQuery table={table_fq} disposition={cfg.bq_write_disposition}")

        # chunked loads in case TS is huge
        first_disp = cfg.bq_write_disposition
        next_disp = "WRITE_APPEND"

        for i in range(0, len(df_ts), int(cfg.write_batch_rows)):
            chunk = df_ts.iloc[i : i + int(cfg.write_batch_rows)].copy()
            _write_df_to_bq(cfg, chunk, table_fq, first_disp)
            console.print(f"[green]✅ Wrote baseline chunk[/green] rows={len(chunk):,} disposition={first_disp}")
            first_disp = next_disp

    return out_path


def load_cfg_from_env(dataset_id: str, feature_version: str) -> BaselineBuildConfig:
    local_root = Path(os.getenv("LOCAL_ROOT", "./data/processed"))

    pitch_types = _parse_csv_list(os.getenv("BASELINE_PITCH_TYPES", "")) or ("Fastball",)
    mode = (os.getenv("BASELINE_MODE", "velocity") or "velocity").strip().lower()
    T = int(os.getenv("TARGET_TIMESTEPS", "700"))
    max_p = int(os.getenv("BASELINE_MAX_PITCHES_PER_GROUP", "50"))

    return BaselineBuildConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        target_timesteps=T,
        mode=mode,
        pitch_types=tuple(pitch_types),
        max_pitches_per_group=max_p,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_src_dataset=(os.getenv("BQ_SRC_DATASET") or dataset_id).strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        pitch_core_table=(os.getenv("BQ_PITCH_CORE_TABLE") or "pitch_core_v1").strip(),
        nf_time_series_table=(os.getenv("BQ_NF_TIME_SERIES_TABLE") or "gold_newtforce_time_series").strip(),
        write_bq=(os.getenv("BQ_WRITE_BASELINES", "0").strip().lower() in ("1", "true", "yes")),
        bq_out_dataset=(os.getenv("BQ_OUT_DATASET") or "analysis_model").strip(),
        bq_write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
        bq_baseline_table_base=(os.getenv("BQ_BASELINE_TABLE_BASE") or "baseline_model").strip(),
        local_root=local_root,
    )
