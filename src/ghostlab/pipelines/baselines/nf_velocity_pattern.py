# src/ghostlab/pipelines/baselines/nf_velocity_pattern.py
"""
Velocity Best Pattern Baseline — completely separate from the accuracy pipeline.

Design:
  - Pool ALL Fastball + Sinker pitches per player (no candidate flag — we compute top % ourselves)
  - Take top TOP_VELOCITY_PCT (default 10%) by RelSpeed per player
  - Build mean NF waveform from those "max effort" pitches
  - pitch_type label = "FB_SI" in all output rows
  - Output → baseline_model_velocity

The resulting pattern answers: "what does YOUR body do when you throw your hardest?"
Every subsequent FB/SI is scored against it to measure how close the delivery came
to that max-effort template.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.interpolate import interp1d
from rich.console import Console

from ghostlab.io.bq import BQConfig, bq_client

console = Console()

TS_FEATURES    = ["fx_lb", "fy_lb", "fz_lb", "x_in", "y_in"]
FORCE_FEATURES = {"fx_lb", "fy_lb", "fz_lb"}

PITCH_TYPE_LABEL = "FB_SI"
BASELINE_KIND    = "best_velocity_pct"
HARD_PITCH_TYPES = ("Fastball", "Sinker")


@dataclass(frozen=True)
class VelocityBaselineConfig:
    # Identity
    dataset_id: str
    feature_version: str

    # Resampling
    target_timesteps: int = 700

    # Candidate selection — top X% of FB+SI by velocity per player
    top_velocity_pct: float = 0.10
    min_pitches_for_pattern: int = 5

    # BQ connection
    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_src_dataset: str = ""
    bq_analysis_dataset: str = "analysis_model"
    pitch_core_table: str = "pitch_core_v1"
    nf_time_series_table: str = "gold_newtforce_time_series"

    # Output
    write_bq: bool = False
    bq_out_dataset: str = "analysis_model"
    bq_write_disposition: str = "WRITE_TRUNCATE"
    bq_baseline_table: str = "baseline_model_velocity"

    local_root: Path = Path("./data/processed")
    write_batch_rows: int = 200_000
    file_batch_size: int = 800


# ── table helpers ─────────────────────────────────────────────────────────────

def _pitch_core_fq(cfg: VelocityBaselineConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{cfg.pitch_core_table}"


def _ts_fq(cfg: VelocityBaselineConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_src_dataset}.{cfg.nf_time_series_table}"


def _out_fq(cfg: VelocityBaselineConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_out_dataset}.{cfg.bq_baseline_table}"


def _make_baseline_id(cfg: VelocityBaselineConfig, player: str) -> str:
    return f"{cfg.dataset_id}|{cfg.feature_version}|{BASELINE_KIND}|{player}|{PITCH_TYPE_LABEL}"


def _dest_dir(cfg: VelocityBaselineConfig) -> Path:
    return cfg.local_root / cfg.dataset_id / "baselines_velocity" / cfg.feature_version


# ── math helpers ──────────────────────────────────────────────────────────────

def _nan_mean_std(values: list) -> Tuple[Optional[float], Optional[float]]:
    arr = np.array(
        [v for v in values if v is not None and np.isfinite(float(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std()) if arr.size > 1 else 0.0


# ── resampling ────────────────────────────────────────────────────────────────

def _resample_one_pitch(
    pitch_ts: pd.DataFrame, weight_lb: float, T: int
) -> Optional[np.ndarray]:
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
        if feat not in pitch_ts.columns:
            return None
        v = pitch_ts[feat].to_numpy(dtype=np.float64)
        if np.all(~np.isfinite(v)):
            return None
        f = interp1d(
            t, v, kind="linear",
            fill_value="extrapolate", bounds_error=False, assume_sorted=True,
        )
        r = f(new_t).astype(np.float32)
        if feat in FORCE_FEATURES:
            out[j, :] = r / float(weight_lb)  # weight-normalize
        else:
            out[j, :] = r - r.mean()           # mean-center path (shape, not position)

    return out if np.isfinite(out).all() else None


# ── BQ I/O ────────────────────────────────────────────────────────────────────

def _fetch_velocity_candidates(cfg: VelocityBaselineConfig) -> pd.DataFrame:
    """
    Fetch ALL Fastball + Sinker pitches with both Trackman and NF time series.
    No candidate flag filtering — top-X% selection is computed here in Python.
    """
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    sql = f"""
    SELECT
      dataset_id,
      feature_version,
      player_full_name,
      TaggedPitchType                          AS pitch_type,
      PitchID,
      nf_file_name,
      SAFE_CAST(player_weight_lb AS FLOAT64)   AS player_weight_lb,
      SAFE_CAST(pitch_velocity   AS FLOAT64)   AS pitch_velocity,
      SAFE_CAST(RelHeight        AS FLOAT64)   AS RelHeight,
      SAFE_CAST(RelSide          AS FLOAT64)   AS RelSide,
      SAFE_CAST(Extension        AS FLOAT64)   AS Extension
    FROM `{_pitch_core_fq(cfg)}`
    WHERE dataset_id        = @dataset_id
      AND feature_version   = @feature_version
      AND has_trackman      = TRUE
      AND has_nf_timeseries = TRUE
      AND nf_file_name      IS NOT NULL
      AND SAFE_CAST(player_weight_lb AS FLOAT64) > 0
      AND SAFE_CAST(pitch_velocity   AS FLOAT64) > 0
      AND TaggedPitchType IN UNNEST(@pitch_types)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ArrayQueryParameter("pitch_types",      "STRING", list(HARD_PITCH_TYPES)),
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    if not df.empty:
        df["pitch_velocity"]   = pd.to_numeric(df["pitch_velocity"],   errors="coerce")
        df["player_weight_lb"] = pd.to_numeric(df["player_weight_lb"], errors="coerce")
        df = df[df["pitch_velocity"].notna()   & (df["pitch_velocity"]   > 0)].copy()
        df = df[df["player_weight_lb"].notna() & (df["player_weight_lb"] > 0)].copy()
    return df


def _fetch_time_series(cfg: VelocityBaselineConfig, file_names: List[str]) -> pd.DataFrame:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    sql = f"""
    SELECT file_name, time_s, fx_lb, fy_lb, fz_lb, x_in, y_in
    FROM `{_ts_fq(cfg)}`
    WHERE file_name IN UNNEST(@file_names)
    ORDER BY file_name, time_s
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("file_names", "STRING", file_names)]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def _write_df_to_bq(
    cfg: VelocityBaselineConfig, df: pd.DataFrame, table_fq: str, disposition: str
) -> None:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    job = client.load_table_from_dataframe(
        df, table_fq,
        job_config=bigquery.LoadJobConfig(write_disposition=disposition),
    )
    job.result()


# ── main ──────────────────────────────────────────────────────────────────────

def build_velocity_baselines(cfg: VelocityBaselineConfig) -> Path:
    out_dir = _dest_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    console.print(
        f"[bold]Velocity Baseline Build[/bold] "
        f"dataset={cfg.dataset_id} version={cfg.feature_version} "
        f"pool={'+'.join(HARD_PITCH_TYPES)} top={cfg.top_velocity_pct:.0%}"
    )
    console.print(f"Run: {run_id}")

    # 1. Fetch all FB+SI pitches with NF data
    candidates = _fetch_velocity_candidates(cfg)
    if candidates.empty:
        raise RuntimeError("No velocity candidates found. Check pitch_core data.")

    n_players = candidates["player_full_name"].nunique()
    console.print(f"Total FB+SI pitches: {len(candidates):,} | players: {n_players:,}")

    # 2. Take top X% by velocity per player (rank-based to avoid pandas 2.x
    #    groupby/apply dropping the groupby key column into the index)
    candidates = candidates.sort_values(
        ["player_full_name", "pitch_velocity"], ascending=[True, False]
    ).reset_index(drop=True)

    top_n_per_player = candidates.groupby("player_full_name")["pitch_velocity"].transform(
        lambda x: max(cfg.min_pitches_for_pattern, int(np.ceil(len(x) * cfg.top_velocity_pct)))
    )
    vrank = candidates.groupby("player_full_name")["pitch_velocity"].rank(
        method="first", ascending=False
    )
    candidates = candidates[vrank <= top_n_per_player].reset_index(drop=True)

    avg_per_player = len(candidates) / max(1, candidates["player_full_name"].nunique())
    console.print(
        f"After top-{cfg.top_velocity_pct:.0%} cut: {len(candidates):,} pitches "
        f"| avg {avg_per_player:.1f} per player"
    )

    # 3. Fetch NF time series in batches
    file_names = candidates["nf_file_name"].dropna().astype(str).unique().tolist()
    ts_parts: List[pd.DataFrame] = []
    for i in range(0, len(file_names), int(cfg.file_batch_size)):
        batch = file_names[i : i + int(cfg.file_batch_size)]
        part = _fetch_time_series(cfg, batch)
        if not part.empty:
            ts_parts.append(part)

    ts = pd.concat(ts_parts, ignore_index=True) if ts_parts else pd.DataFrame()
    console.print(f"Time series rows: {len(ts):,}")
    ts_by_file: Dict[str, pd.DataFrame] = (
        {k: g for k, g in ts.groupby("file_name")} if not ts.empty else {}
    )

    # 4. Build mean waveform per player
    T = int(cfg.target_timesteps)
    time_s_norm = np.linspace(0.0, 1.0, T).astype(np.float32)

    rows_ts: List[dict] = []
    skipped_empty = skipped_bad = skipped_min = 0

    for player, g in candidates.groupby("player_full_name"):
        baseline_id   = _make_baseline_id(cfg, str(player))
        pitch_arrays: List[np.ndarray] = []
        velocities:   List[float]      = []
        releases:     List[dict]       = []

        for r in g.itertuples(index=False):
            fn = str(r.nf_file_name)
            w  = float(r.player_weight_lb) if pd.notna(r.player_weight_lb) else 0.0

            pitch_ts = ts_by_file.get(fn)
            if pitch_ts is None or pitch_ts.empty:
                skipped_empty += 1
                continue

            arr = _resample_one_pitch(pitch_ts, w, T)
            if arr is None:
                skipped_bad += 1
                continue

            pitch_arrays.append(arr)

            v = float(r.pitch_velocity) if pd.notna(r.pitch_velocity) else None
            if v is not None:
                velocities.append(v)

            releases.append({
                "RelHeight": getattr(r, "RelHeight", None),
                "RelSide":   getattr(r, "RelSide",   None),
                "Extension": getattr(r, "Extension", None),
            })

        if len(pitch_arrays) < cfg.min_pitches_for_pattern:
            console.print(
                f"[yellow]  Skip {player}: {len(pitch_arrays)} valid pitches "
                f"< min={cfg.min_pitches_for_pattern}[/yellow]"
            )
            skipped_min += 1
            continue

        mean_wave = np.stack(pitch_arrays, axis=0).mean(axis=0)  # (5, T)
        n_used    = len(pitch_arrays)

        mean_velo, std_velo = _nan_mean_std(velocities)
        mean_rh,   std_rh   = _nan_mean_std([r.get("RelHeight") for r in releases])
        mean_rs,   std_rs   = _nan_mean_std([r.get("RelSide")   for r in releases])
        mean_ext,  std_ext  = _nan_mean_std([r.get("Extension") for r in releases])

        for t_idx in range(T):
            rows_ts.append({
                "baseline_id":       baseline_id,
                "dataset_id":        cfg.dataset_id,
                "feature_version":   cfg.feature_version,
                "baseline_kind":     BASELINE_KIND,
                "player_full_name":  str(player),
                "pitch_type":        PITCH_TYPE_LABEL,
                "timestep":          int(t_idx),
                "time_s":            float(time_s_norm[t_idx]),
                "fx_lb":             float(mean_wave[0, t_idx]),
                "fy_lb":             float(mean_wave[1, t_idx]),
                "fz_lb":             float(mean_wave[2, t_idx]),
                "x_in":              float(mean_wave[3, t_idx]),
                "y_in":              float(mean_wave[4, t_idx]),
                "n_pitches_used":    n_used,
                "mean_velocity":     mean_velo,
                "stddev_velocity":   std_velo,
                "mean_rel_height":   mean_rh,
                "mean_rel_side":     mean_rs,
                "mean_extension":    mean_ext,
                "stddev_rel_height": std_rh,
                "stddev_rel_side":   std_rs,
                "stddev_extension":  std_ext,
                "created_at":        created_at,
                "run_id":            run_id,
            })

    df_ts = pd.DataFrame(rows_ts)
    if df_ts.empty:
        raise RuntimeError("No velocity baselines produced.")

    out_path = out_dir / "nf_velocity_baselines.parquet"
    df_ts.to_parquet(out_path, index=False)

    manifest = {
        "dataset_id":           cfg.dataset_id,
        "feature_version":      cfg.feature_version,
        "baseline_kind":        BASELINE_KIND,
        "pitch_type_label":     PITCH_TYPE_LABEL,
        "top_velocity_pct":     cfg.top_velocity_pct,
        "run_id":               run_id,
        "created_at":           created_at.isoformat(),
        "target_timesteps":     T,
        "channels":             TS_FEATURES,
        "players":              int(df_ts["player_full_name"].nunique()),
        "skipped_empty_ts":     int(skipped_empty),
        "skipped_bad_resample": int(skipped_bad),
        "skipped_min_pitches":  int(skipped_min),
        "output_file":          str(out_path),
    }
    with open(out_dir / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[green]✅ Velocity baselines written:[/green] {out_path}")
    console.print(
        f"Players: {manifest['players']} | "
        f"skipped (min pitches): {skipped_min} | "
        f"skipped (empty ts): {skipped_empty} | "
        f"skipped (bad resample): {skipped_bad}"
    )

    if cfg.write_bq:
        table_fq   = _out_fq(cfg)
        first_disp = cfg.bq_write_disposition
        next_disp  = "WRITE_APPEND"
        console.print(f"Writing to BigQuery: {table_fq}")
        for i in range(0, len(df_ts), int(cfg.write_batch_rows)):
            chunk = df_ts.iloc[i : i + int(cfg.write_batch_rows)].copy()
            _write_df_to_bq(cfg, chunk, table_fq, first_disp)
            console.print(f"[green]✅ Wrote chunk[/green] rows={len(chunk):,} disposition={first_disp}")
            first_disp = next_disp

    return out_path


def load_cfg_from_env(dataset_id: str, feature_version: str) -> VelocityBaselineConfig:
    return VelocityBaselineConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        target_timesteps=int(os.getenv("TARGET_TIMESTEPS", "700")),
        top_velocity_pct=float(os.getenv("VELOCITY_TOP_PCT", "0.10")),
        min_pitches_for_pattern=int(os.getenv("VELOCITY_MIN_PITCHES", "5")),
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_src_dataset=(os.getenv("BQ_SRC_DATASET") or dataset_id).strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        pitch_core_table=(os.getenv("BQ_PITCH_CORE_TABLE") or "pitch_core_v1").strip(),
        nf_time_series_table=(os.getenv("BQ_NF_TIME_SERIES_TABLE") or "gold_newtforce_time_series").strip(),
        write_bq=(os.getenv("BQ_WRITE_BASELINES", "0").strip().lower() in ("1", "true", "yes")),
        bq_out_dataset=(os.getenv("BQ_OUT_DATASET") or "analysis_model").strip(),
        bq_write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
        bq_baseline_table=(os.getenv("BQ_VELOCITY_BASELINE_TABLE") or "baseline_model_velocity").strip(),
        local_root=Path(os.getenv("LOCAL_ROOT", "./data/processed")),
        file_batch_size=int(os.getenv("NF_TS_FILE_BATCH_SIZE", "800")),
        write_batch_rows=int(os.getenv("BQ_BASELINE_WRITE_BATCH_ROWS", "200000")),
    )
