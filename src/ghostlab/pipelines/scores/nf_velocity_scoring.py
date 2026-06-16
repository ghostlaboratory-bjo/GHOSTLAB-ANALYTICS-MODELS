# src/ghostlab/pipelines/scores/nf_velocity_scoring.py
"""
Velocity Pattern Scoring — completely separate from the accuracy scoring pipeline.

Scores every Fastball + Sinker pitch against the player's "max effort" waveform
built by nf_velocity_pattern.py.

Key differences from accuracy scoring:
  - Reads from baseline_model_velocity (hardcoded, no mode-switching)
  - Scores ALL FB+SI pitches (full pool, not just the top-10% used to build the baseline)
  - Baseline lookup is always (player, "FB_SI") — one pattern per player
  - Adds pitch_velocity, baseline_mean_velocity, velocity_vs_baseline to every output row
  - Output → scores_model_velocity
"""
from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.interpolate import interp1d
from rich.console import Console

from ghostlab.io.bq import BQConfig, bq_client

console = Console()

TS_FEATURES    = ["fx_lb", "fy_lb", "fz_lb", "x_in", "y_in"]
FORCE_FEATURES = {"fx_lb", "fy_lb", "fz_lb"}

BASELINE_KIND    = "best_velocity_pct"
PITCH_TYPE_LABEL = "FB_SI"
HARD_PITCH_TYPES = ("Fastball", "Sinker")

_RELEASE_STD_FLOOR = 0.025  # ~0.3 inches — prevents division by near-zero std


@dataclass(frozen=True)
class VelocityScoreConfig:
    dataset_id: str
    feature_version: str

    target_timesteps: int = 700

    # If None we auto-pick the latest run_id from the baseline table
    baseline_run_id: Optional[str] = None

    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_src_dataset: str = ""
    bq_analysis_dataset: str = "analysis_model"
    pitch_core_table: str = "pitch_core_v1"
    nf_time_series_table: str = "gold_newtforce_time_series"
    bq_baseline_table: str = "baseline_model_velocity"

    write_bq: bool = True
    bq_out_dataset: str = "analysis_model"
    bq_write_disposition: str = "WRITE_TRUNCATE"
    bq_scores_table: str = "scores_model_velocity"

    file_batch_size: int = 800
    write_batch_rows: int = 50_000


# ── table helpers ─────────────────────────────────────────────────────────────

def _pitch_core_fq(cfg: VelocityScoreConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{cfg.pitch_core_table}"


def _ts_fq(cfg: VelocityScoreConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_src_dataset}.{cfg.nf_time_series_table}"


def _baseline_fq(cfg: VelocityScoreConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{cfg.bq_baseline_table}"


def _scores_fq(cfg: VelocityScoreConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_out_dataset}.{cfg.bq_scores_table}"


def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


# ── math helpers ──────────────────────────────────────────────────────────────

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt(np.sum(a * a)))
    db = float(np.sqrt(np.sum(b * b)))
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / (da * db))


def _dist_l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _score_similarity(dist_all: float) -> float:
    return float(100.0 * math.exp(-dist_all / 2.0))


def _score_release(
    pitch_rh: Optional[float],
    pitch_rs: Optional[float],
    pitch_ext: Optional[float],
    stats: dict,
) -> Tuple[float, float, float, float]:
    """Returns (release_dist, ext_dev, release_similarity, extension_similarity)."""
    mean_rh  = stats.get("mean_rel_height")
    mean_rs  = stats.get("mean_rel_side")
    mean_ext = stats.get("mean_extension")

    if any(v is None for v in [mean_rh, mean_rs, mean_ext, pitch_rh, pitch_rs, pitch_ext]):
        return 0.0, 0.0, 0.0, 0.0

    std_rh  = max(float(stats.get("stddev_rel_height") or 0.0), _RELEASE_STD_FLOOR)
    std_rs  = max(float(stats.get("stddev_rel_side")   or 0.0), _RELEASE_STD_FLOOR)
    std_ext = max(float(stats.get("stddev_extension")  or 0.0), _RELEASE_STD_FLOOR)

    rh_dev  = abs(float(pitch_rh)  - float(mean_rh))  / std_rh
    rs_dev  = abs(float(pitch_rs)  - float(mean_rs))  / std_rs
    ext_dev = abs(float(pitch_ext) - float(mean_ext)) / std_ext

    release_dist       = math.sqrt(rh_dev ** 2 + rs_dev ** 2)
    release_similarity = 100.0 * math.exp(-release_dist / 2.0)
    extension_sim      = 100.0 * math.exp(-ext_dev     / 2.0)
    return release_dist, ext_dev, release_similarity, extension_sim


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
            v = r / float(weight_lb)
        else:
            v = r - r.mean()           # mean-center path (remove position offset)

        # Z-score normalize so all channels have unit variance before L2 comparison.
        # Must match the same normalization applied in nf_velocity_pattern.py.
        std = float(np.std(v))
        out[j, :] = v / max(std, 1e-6)

    return out if np.isfinite(out).all() else None


# ── BQ helpers ────────────────────────────────────────────────────────────────

def _pick_latest_baseline_run_id(cfg: VelocityScoreConfig) -> str:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    sql = f"""
    SELECT run_id
    FROM `{_baseline_fq(cfg)}`
    WHERE dataset_id      = @dataset_id
      AND feature_version = @feature_version
      AND baseline_kind   = @baseline_kind
    QUALIFY ROW_NUMBER() OVER (ORDER BY created_at DESC) = 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ScalarQueryParameter("baseline_kind",   "STRING", BASELINE_KIND),
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    if df.empty:
        raise RuntimeError(
            f"No velocity baseline rows found in `{_baseline_fq(cfg)}`. "
            "Run velocity-baseline first."
        )
    return str(df.iloc[0]["run_id"])


def _fetch_baseline_meta(cfg: VelocityScoreConfig, baseline_run_id: str) -> pd.DataFrame:
    """One row per player. Includes velocity signature and release point stats."""
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    sql = f"""
    SELECT
      baseline_id,
      ANY_VALUE(player_full_name) AS player_full_name,
      ANY_VALUE(pitch_type)       AS pitch_type,
      MAX(n_pitches_used)         AS n_pitches_used,
      ANY_VALUE(run_id)           AS run_id,
      MAX(created_at)             AS created_at,
      MAX(mean_velocity)          AS mean_velocity,
      MAX(stddev_velocity)        AS stddev_velocity,
      MAX(mean_rel_height)        AS mean_rel_height,
      MAX(mean_rel_side)          AS mean_rel_side,
      MAX(mean_extension)         AS mean_extension,
      MAX(stddev_rel_height)      AS stddev_rel_height,
      MAX(stddev_rel_side)        AS stddev_rel_side,
      MAX(stddev_extension)       AS stddev_extension
    FROM `{_baseline_fq(cfg)}`
    WHERE dataset_id      = @dataset_id
      AND feature_version = @feature_version
      AND baseline_kind   = @baseline_kind
      AND run_id          = @run_id
    GROUP BY baseline_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ScalarQueryParameter("baseline_kind",   "STRING", BASELINE_KIND),
            bigquery.ScalarQueryParameter("run_id",          "STRING", baseline_run_id),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def _fetch_baseline_ts(cfg: VelocityScoreConfig, baseline_run_id: str) -> pd.DataFrame:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    sql = f"""
    SELECT baseline_id, timestep, fx_lb, fy_lb, fz_lb, x_in, y_in
    FROM `{_baseline_fq(cfg)}`
    WHERE dataset_id      = @dataset_id
      AND feature_version = @feature_version
      AND baseline_kind   = @baseline_kind
      AND run_id          = @run_id
    ORDER BY baseline_id, timestep
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ScalarQueryParameter("baseline_kind",   "STRING", BASELINE_KIND),
            bigquery.ScalarQueryParameter("run_id",          "STRING", baseline_run_id),
        ]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


def _fetch_pitches_to_score(cfg: VelocityScoreConfig) -> pd.DataFrame:
    """Full rebuild: ALL FB+SI pitches with NF data — no date filter."""
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    sql = f"""
    SELECT
      dataset_id,
      feature_version,
      PitchID,
      nf_file_name,
      event_ts,
      session_date,
      player_full_name,
      PitcherId,
      PitcherThrows,
      TaggedPitchType,
      player_weight_lb,
      SAFE_CAST(pitch_velocity AS FLOAT64) AS pitch_velocity,
      RelHeight, RelSide, Extension,
      VertRelAngle, HorzRelAngle,
      SpinRate, SpinAxis, Tilt,
      VertBreak, InducedVertBreak, HorzBreak,
      VertApprAngle, HorzApprAngle,
      rel_height_delta, rel_side_delta, extension_delta, relspeed_delta,
      spinrate_delta, vert_rel_angle_delta, horz_rel_angle_delta
    FROM `{_pitch_core_fq(cfg)}`
    WHERE dataset_id         = @dataset_id
      AND has_trackman       = TRUE
      AND has_nf_timeseries  = TRUE
      AND nf_file_name       IS NOT NULL
      AND player_weight_lb   IS NOT NULL
      AND SAFE_CAST(player_weight_lb AS FLOAT64) > 0
      AND TaggedPitchType IN UNNEST(@pitch_types)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id",  "STRING", cfg.dataset_id),
            bigquery.ArrayQueryParameter("pitch_types",  "STRING", list(HARD_PITCH_TYPES)),
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    if df.empty:
        return df
    df["player_weight_lb"] = pd.to_numeric(df["player_weight_lb"], errors="coerce")
    df["pitch_velocity"]   = pd.to_numeric(df["pitch_velocity"],   errors="coerce")
    return df[df["player_weight_lb"].notna() & (df["player_weight_lb"] > 0)].copy()


def _fetch_time_series(cfg: VelocityScoreConfig, file_names: List[str]) -> pd.DataFrame:
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


def _build_baseline_waveforms(ts_df: pd.DataFrame, T: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if ts_df.empty:
        return out
    for baseline_id, g in ts_df.groupby("baseline_id"):
        g = g.sort_values("timestep")
        if len(g) != T:
            continue
        arr = np.stack(
            [
                g["fx_lb"].to_numpy(dtype=np.float32),
                g["fy_lb"].to_numpy(dtype=np.float32),
                g["fz_lb"].to_numpy(dtype=np.float32),
                g["x_in"].to_numpy(dtype=np.float32),
                g["y_in"].to_numpy(dtype=np.float32),
            ],
            axis=0,
        )
        if np.isfinite(arr).all():
            out[str(baseline_id)] = arr
    return out


def _write_df_to_bq(
    cfg: VelocityScoreConfig, df: pd.DataFrame, table_fq: str, disposition: str
) -> None:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    job = client.load_table_from_dataframe(
        df, table_fq,
        job_config=bigquery.LoadJobConfig(write_disposition=disposition),
    )
    job.result()


# ── main ──────────────────────────────────────────────────────────────────────

def score_velocity_pitches(cfg: VelocityScoreConfig) -> Dict[str, int]:
    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    console.print(
        f"[bold]Velocity Scoring[/bold] "
        f"dataset={cfg.dataset_id} version={cfg.feature_version} "
        f"pool={'+'.join(HARD_PITCH_TYPES)}"
    )
    console.print(f"Run: {run_id}")

    # 1. Resolve baseline
    baseline_run_id = cfg.baseline_run_id or _pick_latest_baseline_run_id(cfg)
    console.print(f"Using baseline_run_id={baseline_run_id}")

    meta = _fetch_baseline_meta(cfg, baseline_run_id)
    if meta.empty:
        raise RuntimeError("No velocity baseline meta found. Run velocity-baseline first.")

    # baseline_id keyed by player — one pattern per player (always "FB_SI")
    baseline_id_map: Dict[str, str] = {
        str(r["player_full_name"]): str(r["baseline_id"]) for _, r in meta.iterrows()
    }

    velocity_meta_map: Dict[str, dict] = {
        str(r["baseline_id"]): {
            "mean_velocity":     r.get("mean_velocity"),
            "stddev_velocity":   r.get("stddev_velocity"),
            "mean_rel_height":   r.get("mean_rel_height"),
            "mean_rel_side":     r.get("mean_rel_side"),
            "mean_extension":    r.get("mean_extension"),
            "stddev_rel_height": r.get("stddev_rel_height"),
            "stddev_rel_side":   r.get("stddev_rel_side"),
            "stddev_extension":  r.get("stddev_extension"),
        }
        for _, r in meta.iterrows()
    }

    # 2. Load baseline waveforms
    bts = _fetch_baseline_ts(cfg, baseline_run_id)
    T = int(cfg.target_timesteps)
    baseline_wave = _build_baseline_waveforms(bts, T)
    console.print(f"Baselines loaded: {len(baseline_wave):,} waveforms")

    # 3. Fetch all FB+SI pitches to score
    pitches = _fetch_pitches_to_score(cfg)
    if pitches.empty:
        console.print("[yellow]No eligible pitches to score.[/yellow]")
        return {"pitches": 0, "scored": 0, "skipped_no_baseline": 0, "skipped_bad_ts": 0}

    pitches["baseline_id"] = pitches["player_full_name"].apply(
        lambda p: baseline_id_map.get(str(p))
    )

    skipped_no_baseline = int(pitches["baseline_id"].isna().sum())
    pitches = pitches[pitches["baseline_id"].notna()].copy()
    pitches["baseline_id"] = pitches["baseline_id"].astype(str)

    if pitches.empty:
        console.print("[yellow]All pitches skipped: no baseline found for any player.[/yellow]")
        return {"pitches": 0, "scored": 0, "skipped_no_baseline": skipped_no_baseline, "skipped_bad_ts": 0}

    file_names = pitches["nf_file_name"].dropna().astype(str).unique().tolist()
    console.print(f"Pitches eligible: {len(pitches):,} | unique files: {len(file_names):,}")

    score_fq       = _scores_fq(cfg)
    out_rows:       List[dict] = []
    skipped_bad_ts  = 0
    write_disp      = cfg.bq_write_disposition
    write_disp_next = "WRITE_APPEND"

    pitches_by_file = {fn: g for fn, g in pitches.groupby("nf_file_name")}

    def _to_float(v) -> Optional[float]:
        return float(v) if v is not None and pd.notna(v) else None

    for batch in _chunks(file_names, int(cfg.file_batch_size)):
        ts = _fetch_time_series(cfg, batch)
        if ts.empty:
            continue
        ts_by_file = {k: g for k, g in ts.groupby("file_name")}

        for fn in batch:
            g_pitches = pitches_by_file.get(fn)
            if g_pitches is None or g_pitches.empty:
                continue

            pitch_ts = ts_by_file.get(fn)
            if pitch_ts is None or pitch_ts.empty:
                skipped_bad_ts += int(len(g_pitches))
                continue

            for r in g_pitches.itertuples(index=False):
                w_lb = float(r.player_weight_lb) if pd.notna(r.player_weight_lb) else 0.0
                arr  = _resample_one_pitch(pitch_ts, w_lb, T)
                if arr is None:
                    skipped_bad_ts += 1
                    continue

                bid  = str(r.baseline_id)
                base = baseline_wave.get(bid)
                if base is None:
                    skipped_bad_ts += 1
                    continue

                # Waveform distances
                dist_all    = _dist_l2(arr,        base)
                dist_forces = _dist_l2(arr[0:3, :], base[0:3, :])
                dist_path   = _dist_l2(arr[3:5, :], base[3:5, :])

                corr_ch     = [_safe_corr(arr[i, :], base[i, :]) for i in range(5)]
                corr_forces = float(np.mean(corr_ch[0:3]))
                corr_path   = float(np.mean(corr_ch[3:5]))
                corr_all    = float(np.mean(corr_ch))

                nf_sim = _score_similarity(dist_all)

                # Release point deviation vs max-effort signature
                vel_stats = velocity_meta_map.get(bid, {})
                release_dist, ext_dev, release_sim, extension_sim = _score_release(
                    _to_float(getattr(r, "RelHeight",  None)),
                    _to_float(getattr(r, "RelSide",    None)),
                    _to_float(getattr(r, "Extension",  None)),
                    vel_stats,
                )

                has_release = vel_stats.get("mean_rel_height") is not None
                if has_release:
                    pattern_sim = 0.70 * nf_sim + 0.20 * release_sim + 0.10 * extension_sim
                else:
                    pattern_sim = nf_sim

                # Velocity delta vs the player's max-effort mean
                pitch_velocity    = _to_float(getattr(r, "pitch_velocity", None))
                baseline_mean_vel = _to_float(vel_stats.get("mean_velocity"))
                velocity_vs_baseline = (
                    (pitch_velocity - baseline_mean_vel)
                    if pitch_velocity is not None and baseline_mean_vel is not None
                    else None
                )

                out_rows.append({
                    "dataset_id":              r.dataset_id,
                    "feature_version":         r.feature_version,
                    "PitchID":                 str(r.PitchID),
                    "nf_file_name":            str(r.nf_file_name),
                    "event_ts":                r.event_ts,
                    "session_date":            r.session_date,
                    "player_full_name":        r.player_full_name,
                    "PitcherId":               int(r.PitcherId) if pd.notna(r.PitcherId) else None,
                    "PitcherThrows":           r.PitcherThrows,
                    "TaggedPitchType":         r.TaggedPitchType,
                    "baseline_kind":           BASELINE_KIND,
                    "baseline_id":             bid,
                    "baseline_run_id":         baseline_run_id,
                    "player_weight_lb":        float(w_lb),
                    # Waveform distances
                    "dist_l2_all":             float(dist_all),
                    "dist_l2_forces":          float(dist_forces),
                    "dist_l2_path":            float(dist_path),
                    "corr_all":                float(corr_all),
                    "corr_forces":             float(corr_forces),
                    "corr_path":               float(corr_path),
                    # Similarity scores
                    "similarity_score":        float(nf_sim),
                    "release_dist":            float(release_dist),
                    "ext_dev":                 float(ext_dev),
                    "release_similarity":      float(release_sim),
                    "extension_similarity":    float(extension_sim),
                    "pattern_similarity":      float(pattern_sim),
                    # Velocity coaching columns
                    "pitch_velocity":          pitch_velocity,
                    "baseline_mean_velocity":  baseline_mean_vel,
                    "velocity_vs_baseline":    velocity_vs_baseline,
                    "created_at":              created_at,
                    "run_id":                  run_id,
                })

        if cfg.write_bq and len(out_rows) >= int(cfg.write_batch_rows):
            df_out = pd.DataFrame(out_rows)
            _write_df_to_bq(cfg, df_out, score_fq, write_disp)
            console.print(
                f"[green]✅ Wrote velocity scores[/green] rows={len(df_out):,} "
                f"disposition={write_disp}"
            )
            write_disp = write_disp_next
            out_rows = []

    if cfg.write_bq and out_rows:
        df_out = pd.DataFrame(out_rows)
        _write_df_to_bq(cfg, df_out, score_fq, write_disp)
        console.print(
            f"[green]✅ Wrote velocity scores[/green] rows={len(df_out):,} "
            f"disposition={write_disp}"
        )

    total_scored = int(len(pitches)) - int(skipped_bad_ts)
    console.print(
        f"[bold green]Velocity scoring complete.[/bold green] "
        f"eligible={len(pitches):,} scored={total_scored:,} "
        f"skipped_no_baseline={skipped_no_baseline} "
        f"skipped_bad_ts={skipped_bad_ts}"
    )
    return {
        "pitches":              int(len(pitches)),
        "scored":               total_scored,
        "skipped_no_baseline":  int(skipped_no_baseline),
        "skipped_bad_ts":       int(skipped_bad_ts),
    }


def load_cfg_from_env(dataset_id: str, feature_version: str) -> VelocityScoreConfig:
    return VelocityScoreConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_src_dataset=(os.getenv("BQ_SRC_DATASET") or dataset_id).strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        pitch_core_table=(os.getenv("BQ_PITCH_CORE_TABLE") or "pitch_core_v1").strip(),
        nf_time_series_table=(os.getenv("BQ_NF_TIME_SERIES_TABLE") or "gold_newtforce_time_series").strip(),
        bq_baseline_table=(os.getenv("BQ_VELOCITY_BASELINE_TABLE") or "baseline_model_velocity").strip(),
        write_bq=(os.getenv("BQ_WRITE_SCORES", "0").strip().lower() in ("1", "true", "yes")),
        bq_out_dataset=(os.getenv("BQ_OUT_DATASET") or "analysis_model").strip(),
        bq_write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
        bq_scores_table=(os.getenv("BQ_VELOCITY_SCORES_TABLE") or "scores_model_velocity").strip(),
        file_batch_size=int(os.getenv("NF_TS_FILE_BATCH_SIZE", "800")),
        write_batch_rows=int(os.getenv("BQ_SCORES_WRITE_BATCH_ROWS", "50000")),
    )
