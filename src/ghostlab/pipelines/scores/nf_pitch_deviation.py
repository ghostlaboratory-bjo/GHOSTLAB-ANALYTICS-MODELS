# src/ghostlab/pipelines/scores/nf_pitch_deviation.py
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

TS_FEATURES = ["fx_lb", "fy_lb", "fz_lb", "x_in", "y_in"]
FORCE_FEATURES = {"fx_lb", "fy_lb", "fz_lb"}


@dataclass(frozen=True)
class ScoreNFConfig:
    dataset_id: str
    feature_version: str

    mode: str = "velocity"

    # ✅ Default Fastball only
    pitch_types: Tuple[str, ...] = ("Fastball",)
    target_timesteps: int = 700

    # ✅ FULL REBUILD PHILOSOPHY: no "days_back" filtering in scoring.
    # (Kept out of the config entirely on purpose.)

    # Optional: still allow selecting a specific baseline run; otherwise we pick latest
    baseline_run_id: Optional[str] = None

    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_src_dataset: str = ""
    bq_analysis_dataset: str = "analysis_model"
    pitch_core_table: str = "pitch_core_v1"
    nf_time_series_table: str = "gold_newtforce_time_series"

    # ✅ baselines live in baseline_model_velocity|accuracy
    bq_baseline_table_base: str = "baseline_model"

    # Outputs
    write_bq: bool = True
    bq_out_dataset: str = "analysis_model"
    bq_write_disposition: str = "WRITE_TRUNCATE"  # full rebuild default
    bq_scores_table_base: str = "scores_model"  # -> scores_model_velocity|accuracy

    file_batch_size: int = 800
    write_batch_rows: int = 50_000


def _baseline_kind(mode: str) -> str:
    m = (mode or "").strip().lower()
    return "best_accuracy_pct" if m == "accuracy" else "best_velocity_pct"


def _scores_table_name(cfg: ScoreNFConfig) -> str:
    mode = (cfg.mode or "").strip().lower()
    return f"{cfg.bq_scores_table_base}_{mode}"


def _baseline_table_name(cfg: ScoreNFConfig) -> str:
    mode = (cfg.mode or "").strip().lower()
    return f"{cfg.bq_baseline_table_base}_{mode}"  # baseline_model_velocity|accuracy


def _pitch_core_fq(cfg: ScoreNFConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{cfg.pitch_core_table}"


def _ts_fq(cfg: ScoreNFConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_src_dataset}.{cfg.nf_time_series_table}"


def _baseline_fq(cfg: ScoreNFConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{_baseline_table_name(cfg)}"


def _out_fq(cfg: ScoreNFConfig, table: str) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_out_dataset}.{table}"


def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return 0.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt(np.sum(a * a)))
    db = float(np.sqrt(np.sum(b * b)))
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / (da * db))


def _resample_one_pitch(pitch_ts: pd.DataFrame, weight_lb: float, T: int) -> Optional[np.ndarray]:
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
            # Mean-center path channels so comparison captures movement shape,
            # not absolute plate position (which shifts between sessions).
            out[j, :] = r - r.mean()

    if not np.isfinite(out).all():
        return None
    return out


def _pick_latest_baseline_run_id(cfg: ScoreNFConfig) -> str:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    mode = (cfg.mode or "").strip().lower()
    kind = _baseline_kind(mode)

    sql = f"""
    SELECT run_id
    FROM `{_baseline_fq(cfg)}`
    WHERE dataset_id = @dataset_id
      AND feature_version = @feature_version
      AND mode = @mode
      AND baseline_kind = @baseline_kind
    QUALIFY ROW_NUMBER() OVER (ORDER BY created_at DESC) = 1
    """
    job = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ScalarQueryParameter("mode", "STRING", mode),
            bigquery.ScalarQueryParameter("baseline_kind", "STRING", kind),
        ]
    )
    df = client.query(sql, job_config=job).to_dataframe()
    if df.empty:
        raise RuntimeError(
            f"No baseline rows found in `{_baseline_fq(cfg)}` for mode={mode} kind={kind}. Build baselines first."
        )
    return str(df.iloc[0]["run_id"])


def _fetch_baseline_meta(cfg: ScoreNFConfig, baseline_run_id: str) -> pd.DataFrame:
    """
    Meta is derived from the baseline table itself (one row per baseline_id).
    Includes release point stats (scalars repeated across all waveform rows).
    """
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    mode = (cfg.mode or "").strip().lower()
    kind = _baseline_kind(mode)

    sql = f"""
    SELECT
      baseline_id,
      ANY_VALUE(player_full_name) AS player_full_name,
      ANY_VALUE(pitch_type)       AS pitch_type,
      MAX(n_pitches_used)         AS n_pitches_used,
      ANY_VALUE(run_id)           AS run_id,
      MAX(created_at)             AS created_at,
      MAX(mean_rel_height)        AS mean_rel_height,
      MAX(mean_rel_side)          AS mean_rel_side,
      MAX(mean_extension)         AS mean_extension,
      MAX(stddev_rel_height)      AS stddev_rel_height,
      MAX(stddev_rel_side)        AS stddev_rel_side,
      MAX(stddev_extension)       AS stddev_extension
    FROM `{_baseline_fq(cfg)}`
    WHERE dataset_id = @dataset_id
      AND feature_version = @feature_version
      AND mode = @mode
      AND baseline_kind = @baseline_kind
      AND run_id = @run_id
      AND pitch_type IN UNNEST(@pitch_types)
    GROUP BY baseline_id
    """
    # Include "ALL" so cross-pitch-type accuracy baselines (pitch_type="ALL") are found
    pitch_types_param = list(cfg.pitch_types) + ["ALL"]
    job = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ScalarQueryParameter("mode", "STRING", mode),
            bigquery.ScalarQueryParameter("baseline_kind", "STRING", kind),
            bigquery.ScalarQueryParameter("run_id", "STRING", baseline_run_id),
            bigquery.ArrayQueryParameter("pitch_types", "STRING", pitch_types_param),
        ]
    )
    return client.query(sql, job_config=job).to_dataframe()


def _fetch_baseline_ts(cfg: ScoreNFConfig, baseline_run_id: str) -> pd.DataFrame:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    mode = (cfg.mode or "").strip().lower()
    kind = _baseline_kind(mode)

    sql = f"""
    SELECT
      baseline_id,
      timestep,
      fx_lb,
      fy_lb,
      fz_lb,
      x_in,
      y_in
    FROM `{_baseline_fq(cfg)}`
    WHERE dataset_id = @dataset_id
      AND feature_version = @feature_version
      AND mode = @mode
      AND baseline_kind = @baseline_kind
      AND run_id = @run_id
    ORDER BY baseline_id, timestep
    """
    job = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ScalarQueryParameter("mode", "STRING", mode),
            bigquery.ScalarQueryParameter("baseline_kind", "STRING", kind),
            bigquery.ScalarQueryParameter("run_id", "STRING", baseline_run_id),
        ]
    )
    return client.query(sql, job_config=job).to_dataframe()


def _fetch_pitches_to_score(cfg: ScoreNFConfig) -> pd.DataFrame:
    """
    FULL rebuild: fetch ALL eligible pitches (no days_back filter).
    Note: pitch_types still applies (by design).
    """
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))

    params: List[bigquery.QueryParameter] = [
        bigquery.ScalarQueryParameter("dataset_id", "STRING", cfg.dataset_id),
        bigquery.ArrayQueryParameter("pitch_types", "STRING", list(cfg.pitch_types)),
    ]

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
      pitch_velocity,
      miss_distance_in,
      is_strike_target,
      RelHeight, RelSide, Extension, VertRelAngle, HorzRelAngle,
      SpinRate, SpinAxis, Tilt,
      VertBreak, InducedVertBreak, HorzBreak,
      VertApprAngle, HorzApprAngle,
      rel_height_delta, rel_side_delta, extension_delta, relspeed_delta,
      spinrate_delta, vert_rel_angle_delta, horz_rel_angle_delta
    FROM `{_pitch_core_fq(cfg)}`
    WHERE dataset_id = @dataset_id
      AND has_trackman = TRUE
      AND has_nf_timeseries = TRUE
      AND nf_file_name IS NOT NULL
      AND player_weight_lb IS NOT NULL
      AND SAFE_CAST(player_weight_lb AS FLOAT64) > 0
      AND TaggedPitchType IN UNNEST(@pitch_types)
    """
    df = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    if df.empty:
        return df
    df["player_weight_lb"] = pd.to_numeric(df["player_weight_lb"], errors="coerce")
    return df[df["player_weight_lb"].notna() & (df["player_weight_lb"] > 0)].copy()


def _fetch_time_series(cfg: ScoreNFConfig, file_names: List[str]) -> pd.DataFrame:
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
    job = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("file_names", "STRING", file_names)])
    return client.query(sql, job_config=job).to_dataframe()


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


def _features_from_wave(w: np.ndarray) -> Dict[str, float]:
    c, t = w.shape
    assert c == 5
    tt = np.linspace(0.0, 1.0, t, dtype=np.float64)
    fx, fy, fz, x, y = [w[i, :].astype(np.float64) for i in range(5)]

    def peak(v: np.ndarray):
        idx = int(np.argmax(v))
        return float(v[idx]), float(tt[idx])

    pfx, pfx_t = peak(fx)
    pfy, pfy_t = peak(fy)
    pfz, pfz_t = peak(fz)

    if t >= 2:
        ifx = float(np.trapezoid(fx, tt))
        ify = float(np.trapezoid(fy, tt))
        ifz = float(np.trapezoid(fz, tt))
    else:
        ifx = ify = ifz = 0.0

    return {
        "peak_fx_lb": pfx,
        "peak_fy_lb": pfy,
        "peak_fz_lb": pfz,
        "peak_fx_t": pfx_t,
        "peak_fy_t": pfy_t,
        "peak_fz_t": pfz_t,
        "impulse_fx": ifx,
        "impulse_fy": ify,
        "impulse_fz": ifz,
        "path_x_range_in": float(np.nanmax(x) - np.nanmin(x)),
        "path_y_range_in": float(np.nanmax(y) - np.nanmin(y)),
    }


def _dist_l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _score_similarity(dist_all: float) -> float:
    return float(100.0 * math.exp(-dist_all / 2.0))


_RELEASE_STD_FLOOR = 0.025  # ~0.3 inches, prevents division by near-zero std


def _score_release(
    pitch_rh: Optional[float],
    pitch_rs: Optional[float],
    pitch_ext: Optional[float],
    stats: dict,
) -> Tuple[float, float, float, float]:
    """
    Returns (release_dist, ext_dev, release_similarity, extension_similarity).
    All similarity scores in [0, 100]. Returns zeros if stats or pitch values are missing.
    """
    mean_rh = stats.get("mean_rel_height")
    mean_rs = stats.get("mean_rel_side")
    mean_ext = stats.get("mean_extension")

    if any(v is None for v in [mean_rh, mean_rs, mean_ext, pitch_rh, pitch_rs, pitch_ext]):
        return 0.0, 0.0, 0.0, 0.0

    std_rh = max(float(stats.get("stddev_rel_height") or 0.0), _RELEASE_STD_FLOOR)
    std_rs = max(float(stats.get("stddev_rel_side") or 0.0), _RELEASE_STD_FLOOR)
    std_ext = max(float(stats.get("stddev_extension") or 0.0), _RELEASE_STD_FLOOR)

    rh_dev = abs(float(pitch_rh) - float(mean_rh)) / std_rh
    rs_dev = abs(float(pitch_rs) - float(mean_rs)) / std_rs
    ext_dev = abs(float(pitch_ext) - float(mean_ext)) / std_ext

    release_dist = math.sqrt(rh_dev ** 2 + rs_dev ** 2)
    release_similarity = 100.0 * math.exp(-release_dist / 2.0)
    extension_similarity = 100.0 * math.exp(-ext_dev / 2.0)

    return release_dist, ext_dev, release_similarity, extension_similarity


def _write_df_to_bq(cfg: ScoreNFConfig, df: pd.DataFrame, table_fq: str, disposition: str) -> None:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    job = client.load_table_from_dataframe(
        df,
        table_fq,
        job_config=bigquery.LoadJobConfig(write_disposition=disposition),
    )
    job.result()


def score_nf_pitches(cfg: ScoreNFConfig) -> Dict[str, int]:
    created_at = datetime.now(timezone.utc)

    # NOTE: you said you don't care about run_id if you always TRUNCATE.
    # Keeping it here for now because your tables still have these columns.
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    mode = (cfg.mode or "").strip().lower()
    baseline_kind = _baseline_kind(mode)

    console.print(f"[bold]NF scoring[/bold] dataset={cfg.dataset_id} version={cfg.feature_version} mode={mode}")
    console.print(f"Run: {run_id}")

    baseline_run_id = cfg.baseline_run_id or _pick_latest_baseline_run_id(cfg)
    console.print(f"Using baseline_run_id={baseline_run_id} baseline_kind={baseline_kind}")

    meta = _fetch_baseline_meta(cfg, baseline_run_id)
    if meta.empty:
        raise RuntimeError("No baseline meta found for the selected baseline_run_id.")

    baseline_id_map: Dict[Tuple[str, str], str] = {
        (str(r["player_full_name"]), str(r["pitch_type"])): str(r["baseline_id"]) for _, r in meta.iterrows()
    }

    # Release stats keyed by baseline_id — used for per-pitch release deviation scoring
    release_stats_map: Dict[str, dict] = {
        str(r["baseline_id"]): {
            "mean_rel_height": r.get("mean_rel_height"),
            "mean_rel_side": r.get("mean_rel_side"),
            "mean_extension": r.get("mean_extension"),
            "stddev_rel_height": r.get("stddev_rel_height"),
            "stddev_rel_side": r.get("stddev_rel_side"),
            "stddev_extension": r.get("stddev_extension"),
        }
        for _, r in meta.iterrows()
    }

    def _lookup_baseline_id(player: str, pitch_type: str) -> Optional[str]:
        # Exact match (velocity mode, per pitch type) → fall back to "ALL" (accuracy mode, per player)
        return baseline_id_map.get((player, pitch_type)) or baseline_id_map.get((player, "ALL"))

    bts = _fetch_baseline_ts(cfg, baseline_run_id)
    T = int(cfg.target_timesteps)
    baseline_wave = _build_baseline_waveforms(bts, T)
    console.print(f"Baselines loaded: {len(baseline_wave):,} waveforms")

    pitches = _fetch_pitches_to_score(cfg)
    if pitches.empty:
        console.print("[yellow]No eligible pitches to score.[/yellow]")
        return {"pitches": 0, "scored": 0, "skipped_no_baseline": 0, "skipped_bad_ts": 0}

    pitches["pitch_type"] = pitches["TaggedPitchType"].astype(str)
    pitches["baseline_id"] = pitches.apply(
        lambda r: _lookup_baseline_id(str(r["player_full_name"]), str(r["pitch_type"])),
        axis=1,
    )

    skipped_no_baseline = int(pitches["baseline_id"].isna().sum())
    pitches = pitches[pitches["baseline_id"].notna()].copy()
    pitches["baseline_id"] = pitches["baseline_id"].astype(str)

    if pitches.empty:
        console.print("[yellow]All pitches skipped: no baseline_id found.[/yellow]")
        return {"pitches": 0, "scored": 0, "skipped_no_baseline": skipped_no_baseline, "skipped_bad_ts": 0}

    file_names = pitches["nf_file_name"].dropna().astype(str).unique().tolist()
    console.print(f"Pitches eligible: {len(pitches):,} | unique files: {len(file_names):,}")

    scores_table = _scores_table_name(cfg)
    score_fq = _out_fq(cfg, scores_table)

    out_rows: List[dict] = []
    skipped_bad_ts = 0

    write_disp_first = cfg.bq_write_disposition
    write_disp_next = "WRITE_APPEND"  # still used within a single run for chunked loads

    pitches_by_file = {fn: g for fn, g in pitches.groupby("nf_file_name")}

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
                arr = _resample_one_pitch(pitch_ts, w_lb, T)
                if arr is None:
                    skipped_bad_ts += 1
                    continue

                bid = str(r.baseline_id)
                base = baseline_wave.get(bid)
                if base is None:
                    skipped_bad_ts += 1
                    continue

                dist_all = _dist_l2(arr, base)
                dist_forces = _dist_l2(arr[0:3, :], base[0:3, :])
                dist_path = _dist_l2(arr[3:5, :], base[3:5, :])

                corr_ch = [_safe_corr(arr[i, :], base[i, :]) for i in range(5)]
                corr_forces = float(np.mean(corr_ch[0:3]))
                corr_path = float(np.mean(corr_ch[3:5]))
                corr_all = float(np.mean(corr_ch))

                nf_sim = _score_similarity(dist_all)

                # Release point + extension deviation vs pattern signature
                def _to_float(v) -> Optional[float]:
                    return float(v) if v is not None and pd.notna(v) else None

                rel_stats = release_stats_map.get(bid, {})
                release_dist, ext_dev, release_sim, extension_sim = _score_release(
                    _to_float(getattr(r, "RelHeight", None)),
                    _to_float(getattr(r, "RelSide", None)),
                    _to_float(getattr(r, "Extension", None)),
                    rel_stats,
                )

                has_release = rel_stats.get("mean_rel_height") is not None
                if has_release:
                    # 70% NF body mechanics, 20% release point, 10% extension
                    pattern_sim = 0.70 * nf_sim + 0.20 * release_sim + 0.10 * extension_sim
                else:
                    pattern_sim = nf_sim

                out_rows.append(
                    {
                        "dataset_id": r.dataset_id,
                        "feature_version": r.feature_version,
                        "PitchID": str(r.PitchID),
                        "nf_file_name": str(r.nf_file_name),
                        "event_ts": r.event_ts,
                        "session_date": r.session_date,
                        "player_full_name": r.player_full_name,
                        "PitcherId": int(r.PitcherId) if pd.notna(r.PitcherId) else None,
                        "PitcherThrows": r.PitcherThrows,
                        "TaggedPitchType": r.TaggedPitchType,
                        "mode": mode,
                        "baseline_kind": baseline_kind,
                        "baseline_id": bid,
                        "baseline_run_id": baseline_run_id,
                        "player_weight_lb": float(w_lb),
                        "dist_l2_all": float(dist_all),
                        "dist_l2_forces": float(dist_forces),
                        "dist_l2_path": float(dist_path),
                        "corr_all": float(corr_all),
                        "corr_forces": float(corr_forces),
                        "corr_path": float(corr_path),
                        "similarity_score": float(nf_sim),       # pure NF waveform
                        "release_dist": float(release_dist),      # normalized release deviation
                        "ext_dev": float(ext_dev),                # normalized extension deviation
                        "release_similarity": float(release_sim),
                        "extension_similarity": float(extension_sim),
                        "pattern_similarity": float(pattern_sim), # composite score
                        "created_at": created_at,
                        "run_id": run_id,
                    }
                )

        if cfg.write_bq and len(out_rows) >= int(cfg.write_batch_rows):
            df_out = pd.DataFrame(out_rows)
            _write_df_to_bq(cfg, df_out, score_fq, write_disp_first)
            console.print(f"[green]✅ Wrote scores[/green] rows={len(df_out):,} disposition={write_disp_first}")
            write_disp_first = write_disp_next
            out_rows = []

    if cfg.write_bq and out_rows:
        df_out = pd.DataFrame(out_rows)
        _write_df_to_bq(cfg, df_out, score_fq, write_disp_first)
        console.print(f"[green]✅ Wrote scores[/green] rows={len(df_out):,} disposition={write_disp_first}")

    return {
        "pitches": int(len(pitches)),
        "scored": int(len(pitches)) - int(skipped_bad_ts),
        "skipped_no_baseline": int(skipped_no_baseline),
        "skipped_bad_ts": int(skipped_bad_ts),
    }


def load_cfg_from_env(dataset_id: str, feature_version: str) -> ScoreNFConfig:
    return ScoreNFConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_src_dataset=(os.getenv("BQ_SRC_DATASET") or dataset_id).strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        pitch_core_table=(os.getenv("BQ_PITCH_CORE_TABLE") or "pitch_core_v1").strip(),
        nf_time_series_table=(os.getenv("BQ_NF_TIME_SERIES_TABLE") or "gold_newtforce_time_series").strip(),
        bq_baseline_table_base=(os.getenv("BQ_BASELINE_TABLE_BASE") or "baseline_model").strip(),
        bq_out_dataset=(os.getenv("BQ_OUT_DATASET") or "analysis_model").strip(),
        bq_write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
        bq_scores_table_base=(os.getenv("BQ_SCORES_TABLE_BASE") or "scores_model").strip(),
    )