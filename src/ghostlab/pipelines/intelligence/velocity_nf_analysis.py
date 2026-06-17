# src/ghostlab/pipelines/intelligence/velocity_nf_analysis.py
"""
Velocity NF Correlation Analysis — per-player feature-velocity correlations.

Coach brief:
  Use Trackman FB/SI/CU velocity as the outcome variable. Find the specific
  NewtForce metrics with the highest correlation to each individual's highest
  output. Focus on total force output AND force transfer metrics.

Pipeline:
  1. Fetch all scored FB/SI/Cutter pitches from scores_model_velocity
     (already has pitch_velocity + nf_file_name).
  2. Pull raw NF time series for each pitch.
  3. Compute ~12 per-pitch aggregate NF features:
       Total Force Output: peak_resultant_force, avg_resultant_force,
                           force_impulse, peak_fx, peak_fy, peak_fz
       Transfer Metrics:   time_to_peak_force, late_force_ratio,
                           force_at_late, peak_path_speed,
                           path_arc_length, force_buildup_rate
  4. Per player (min 15 pitches), run Pearson correlations of each feature
     vs pitch_velocity.
  5. Store top-ranked features + coaching text in velocity_nf_analysis_v1.

Output → velocity_nf_analysis_v1
  One row per player × feature (all features stored so UI can choose which
  to surface). Also stores high/low velocity split means for each feature
  so the UI can show concrete "hard throw vs soft throw" numbers.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from rich.console import Console
from scipy.stats import pearsonr

from ghostlab.io.bq import BQConfig, bq_client

console = Console()

# ── Pitch types to include (must match velocity scoring) ─────────────────────
HARD_PITCH_TYPES = ("Fastball", "Sinker", "Cutter")

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURE_META: Dict[str, dict] = {
    # Total Force Output
    "peak_resultant_force": {
        "label":    "Peak Total Force Output",
        "category": "Force Output",
        "unit":     "lb/lb",
        "coaching": (
            "The highest combined force (all axes) produced at any moment during the delivery, "
            "normalized by body weight. Higher = more explosive force generation."
        ),
    },
    "avg_resultant_force": {
        "label":    "Average Force Output",
        "category": "Force Output",
        "unit":     "lb/lb",
        "coaching": (
            "Average total force sustained across the entire delivery. "
            "Reflects overall force production, not just peak explosiveness."
        ),
    },
    "force_impulse": {
        "label":    "Force Impulse (Total Force Applied)",
        "category": "Force Output",
        "unit":     "lb·s/lb",
        "coaching": (
            "The integral of force over time — total force actually applied to propel the ball. "
            "A higher impulse means more force was sustained throughout the delivery."
        ),
    },
    "peak_fx": {
        "label":    "Peak Lateral Force",
        "category": "Force Output",
        "unit":     "lb/lb",
        "coaching": (
            "Peak side-to-side force component. Reflects hip/torso rotation force."
        ),
    },
    "peak_fy": {
        "label":    "Peak Forward Force",
        "category": "Force Output",
        "unit":     "lb/lb",
        "coaching": (
            "Peak forward-directed force. Reflects drive off the rubber and momentum into the pitch."
        ),
    },
    "peak_fz": {
        "label":    "Peak Vertical Force",
        "category": "Force Output",
        "unit":     "lb/lb",
        "coaching": (
            "Peak vertical force component. Reflects how much downward/upward force is generated."
        ),
    },
    # Transfer Metrics
    "time_to_peak_force": {
        "label":    "Force Peak Timing",
        "category": "Transfer",
        "unit":     "% of delivery",
        "coaching": (
            "When in the delivery peak force occurs, normalized 0–100%. "
            "Later peak timing typically means better kinetic chain sequencing — "
            "force is built up and transferred efficiently rather than peaking too early."
        ),
    },
    "late_force_ratio": {
        "label":    "Force Sustained Through Release",
        "category": "Transfer",
        "unit":     "%",
        "coaching": (
            "Ratio of force in the final 20% of delivery to peak force. "
            "A higher ratio means the pitcher 'finishes' well — maintaining force through release "
            "rather than decelerating before the ball leaves the hand."
        ),
    },
    "force_at_late": {
        "label":    "Late-Phase Force Output",
        "category": "Transfer",
        "unit":     "lb/lb",
        "coaching": (
            "Average force in the final 20% of the delivery window. "
            "Directly measures whether force is carried through to ball release."
        ),
    },
    "peak_path_speed": {
        "label":    "Peak Movement Speed",
        "category": "Transfer",
        "unit":     "in/s",
        "coaching": (
            "Maximum speed of the NewtForce sensor movement (from path channels). "
            "Captures how fast the body is moving at its fastest point in the delivery."
        ),
    },
    "path_arc_length": {
        "label":    "Total Delivery Arc Length",
        "category": "Transfer",
        "unit":     "in",
        "coaching": (
            "Total distance the NewtForce sensor travels through the delivery. "
            "Reflects range of motion — longer arc often indicates better hip/shoulder separation."
        ),
    },
    "force_buildup_rate": {
        "label":    "Force Buildup Rate",
        "category": "Transfer",
        "unit":     "lb/lb/s",
        "coaching": (
            "How quickly force builds from start to peak. "
            "A higher rate indicates a more explosive, fast-twitch force generation pattern."
        ),
    },
    # ── NewtForce-named metrics (official taxonomy, validated against velocity) ──
    # Back leg = first move through front foot touch (Fy > 0). Front leg = touch
    # through release (Fy < 0). Peer-study correlations to FB velocity (r):
    # Y Back Score 0.78, YZ Back Score 0.78, Y Front Score 0.75, YZ Front Score 0.71.
    "y_back": {
        "label": "Y Back", "category": "NewtForce Back Leg", "unit": "lbf",
        "coaching": "Back leg peak force toward 2nd base (drive-leg push).",
    },
    "y_back_score": {
        "label": "Y Back Score", "category": "NewtForce Back Leg", "unit": "lbf/lb",
        "coaching": (
            "Y Back normalized by body weight. NewtForce's own pilot study found this the single "
            "strongest velocity predictor (r~0.78). If below peer, check Z Back Score — a pitcher "
            "may be pushing 'into the ground' rather than toward the plate."
        ),
    },
    "z_back": {
        "label": "Z Back", "category": "NewtForce Back Leg", "unit": "lbf",
        "coaching": "Back leg peak vertical force.",
    },
    "z_back_score": {
        "label": "Z Back Score", "category": "NewtForce Back Leg", "unit": "lbf/lb",
        "coaching": "Z Back normalized by body weight. If below peer, check Y Back Score for a 'jumping' pattern with little ground purchase.",
    },
    "yz_back_score": {
        "label": "YZ Back Score", "category": "NewtForce Back Leg", "unit": "lbf/lb",
        "coaching": "Sum of Y Back + Z Back, normalized by body weight. Second-strongest velocity predictor in NewtForce's pilot study (r~0.78).",
    },
    "x_y_back": {
        "label": "X-Y Back", "category": "NewtForce Back Leg", "unit": "lbf",
        "coaching": "Back leg lateral (X) force at the moment of peak Y Back. The most nuanced of the back-leg forces.",
    },
    "y_front": {
        "label": "Y Front", "category": "NewtForce Front Leg", "unit": "lbf",
        "coaching": "Front leg peak braking force toward home plate (magnitude of the deceleration spike).",
    },
    "y_front_score": {
        "label": "Y Front Score", "category": "NewtForce Front Leg", "unit": "lbf/lb",
        "coaching": "Y Front normalized by body weight. Third-strongest velocity predictor (r~0.75). If below peer, interrogate the back leg — great brakes from a poor accelerator is rare.",
    },
    "z_front": {
        "label": "Z Front", "category": "NewtForce Front Leg", "unit": "lbf",
        "coaching": "Front leg peak vertical force (landing-leg block).",
    },
    "z_front_score": {
        "label": "Z Front Score", "category": "NewtForce Front Leg", "unit": "lbf/lb",
        "coaching": "Z Front normalized by body weight.",
    },
    "yz_front_score": {
        "label": "YZ Front Score", "category": "NewtForce Front Leg", "unit": "lbf/lb",
        "coaching": "Sum of Y Front + Z Front, normalized by body weight. Fourth-strongest velocity predictor (r~0.71).",
    },
    "x_y_front": {
        "label": "X-Y Front", "category": "NewtForce Front Leg", "unit": "lbf",
        "coaching": "Front leg lateral (X) force at the moment of peak Y Front.",
    },
    "accel_impulse": {
        "label": "Accel Impulse", "category": "NewtForce Impulse", "unit": "lbf-s",
        "coaching": "Area under the back-leg Y curve — total drive-phase force applied toward the plate. NewtForce's #1 velocity attractor across youth-to-MLB pitchers.",
    },
    "accel_impulse_score": {
        "label": "Accel Impulse Score", "category": "NewtForce Impulse", "unit": "lbf-s/lb",
        "coaching": "Accel Impulse normalized by body weight — the weight-independent version of NewtForce's top velocity attractor.",
    },
    "decel_impulse": {
        "label": "Decel Impulse", "category": "NewtForce Impulse", "unit": "lbf-s",
        "coaching": "Area above the front-leg Y curve — total braking force absorbed during the block phase.",
    },
    "decel_impulse_score": {
        "label": "Decel Impulse Score", "category": "NewtForce Impulse", "unit": "lbf-s/lb",
        "coaching": "Decel Impulse normalized by body weight.",
    },
    "impulse_ratio": {
        "label": "Impulse Ratio", "category": "NewtForce Impulse", "unit": "ratio",
        "coaching": "Accel Impulse Score / Decel Impulse Score — the balance between drive and brake. NewtForce's elite movers tend to converge near ~1.3-1.5.",
    },
    "z_transfer": {
        "label": "Z Transfer", "category": "NewtForce Timing", "unit": "s",
        "coaching": "Time between back-leg peak Z and front-leg peak Z. Generally, faster (smaller) is a sign of cleaner mechanics.",
    },
    "y_transfer": {
        "label": "Y Transfer", "category": "NewtForce Timing", "unit": "s",
        "coaching": "Time between back-leg peak Y and front-leg peak Y.",
    },
    "yz_transfer_back": {
        "label": "YZ Transfer Back", "category": "NewtForce Timing", "unit": "s",
        "coaching": "Time between back-leg peak Y and back-leg peak Z. Bigger numbers are generally worse — anything over 0.10s is a red flag.",
    },
    "yz_transfer_front": {
        "label": "YZ Transfer Front", "category": "NewtForce Timing", "unit": "s",
        "coaching": "Time between front-leg peak Y and front-leg peak Z.",
    },
    "clawback": {
        "label": "Clawback", "category": "NewtForce Timing", "unit": "s",
        "coaching": "Time from front-leg peak Y back to the next local peak of Y — how quickly forward momentum is arrested. Some trainers target lower Clawback; late relievers trend lower than starters.",
    },
    "decel_width": {
        "label": "Decel Width", "category": "NewtForce Timing", "unit": "s",
        "coaching": "Front-leg Y curve width at 75% of front-leg peak Y — the shape of the bottom of the deceleration trough.",
    },
    "stride": {
        "label": "Stride", "category": "NewtForce Stride", "unit": "in",
        "coaching": "Linear distance between back-leg CoP (at back-leg peak Z) and front-leg CoP (at front-leg peak Z).",
    },
    "stride_angle": {
        "label": "Stride Angle", "category": "NewtForce Stride", "unit": "deg",
        "coaching": "Angle of the stride line relative to the mound-to-plate axis. Elite pitchers more commonly land slightly closed, though variance is high.",
    },
}

FEATURE_NAMES = list(FEATURE_META.keys())

MIN_PITCHES = 15  # minimum per player for meaningful correlations


@dataclass(frozen=True)
class VelocityNFAnalysisConfig:
    dataset_id: str
    feature_version: str

    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_analysis_dataset: str = "analysis_model"
    bq_src_dataset: str = ""

    velocity_scores_table: str = "scores_model_velocity"
    nf_time_series_table: str = "gold_newtforce_time_series"
    output_table: str = "velocity_nf_analysis_v1"

    write_disposition: str = "WRITE_TRUNCATE"
    file_batch_size: int = 600


def _analysis(cfg: VelocityNFAnalysisConfig, table: str) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{table}"


def _ts_fq(cfg: VelocityNFAnalysisConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_src_dataset}.{cfg.nf_time_series_table}"


# ── NewtForce-named metric computation ────────────────────────────────────────

def _scan_idx(arr: np.ndarray, start: int, step: int, keep_going) -> int:
    """Advance/retreat from start while keep_going(arr[idx]) holds. Returns last valid idx."""
    idx = start
    n = len(arr)
    while True:
        nxt = idx + step
        if nxt < 0 or nxt >= n or not keep_going(arr[nxt]):
            return idx
        idx = nxt


def _compute_named_nf_metrics(
    t: np.ndarray, raw_fx: np.ndarray, raw_fy: np.ndarray, raw_fz: np.ndarray,
    x: np.ndarray, y: np.ndarray, weight_lb: float,
) -> Optional[dict]:
    """
    Detect back-leg/front-leg phases from the Fy curve and derive the official
    NewtForce-named metrics (Y/Z Back/Front (Score), Accel/Decel Impulse, transfer
    timings, Clawback, Decel Width, Stride, Stride Angle). Returns None if the
    pitch doesn't show a clean back-leg-push / front-leg-block Fy pattern.
    """
    n = len(t)
    if n < 10:
        return None

    idx_y_back = int(np.argmax(raw_fy))
    if raw_fy[idx_y_back] <= 0:
        return None  # no real drive-leg push detected

    # Back-leg positive region: scan outward from the peak while Fy stays positive.
    entry_idx = _scan_idx(raw_fy, idx_y_back, -1, lambda v: v > 0)
    touch_idx = _scan_idx(raw_fy, idx_y_back, +1, lambda v: v > 0)
    touch_idx = min(touch_idx + 1, n - 1)  # first index at/after the zero-crossing

    front_fy = raw_fy[touch_idx:]
    if front_fy.size < 3:
        return None
    idx_y_front = touch_idx + int(np.argmin(front_fy))
    if raw_fy[idx_y_front] >= 0:
        return None  # no real braking phase detected

    release_idx = _scan_idx(raw_fy, idx_y_front, +1, lambda v: v < 0)
    release_idx = min(release_idx + 1, n - 1)

    idx_z_back = int(np.argmax(raw_fz[: touch_idx + 1]))
    idx_z_front = touch_idx + int(np.argmax(raw_fz[touch_idx:]))

    w = max(weight_lb, 1e-6)

    y_back = float(raw_fy[idx_y_back])
    z_back = float(raw_fz[idx_z_back])
    y_front = float(abs(raw_fy[idx_y_front]))
    z_front = float(raw_fz[idx_z_front])

    accel_impulse = float(np.trapezoid(
        np.clip(raw_fy[entry_idx:touch_idx + 1], 0, None), t[entry_idx:touch_idx + 1]
    ))
    decel_impulse = float(np.trapezoid(
        -np.clip(raw_fy[touch_idx:release_idx + 1], None, 0), t[touch_idx:release_idx + 1]
    ))
    accel_impulse_score = accel_impulse / w
    decel_impulse_score = decel_impulse / w
    impulse_ratio = (accel_impulse_score / decel_impulse_score) if decel_impulse_score > 1e-9 else None

    # Clawback: from the front-leg trough, advance while Fy is non-decreasing
    # (riding back up to its next local peak).
    clawback_idx = idx_y_front
    while clawback_idx < n - 1 and raw_fy[clawback_idx + 1] >= raw_fy[clawback_idx]:
        clawback_idx += 1
    clawback = float(t[clawback_idx] - t[idx_y_front])

    # Decel width: full width of the front-leg trough at 75% of its depth.
    threshold = 0.75 * raw_fy[idx_y_front]
    left_idx = _scan_idx(raw_fy, idx_y_front, -1, lambda v: v <= threshold)
    right_idx = _scan_idx(raw_fy, idx_y_front, +1, lambda v: v <= threshold)
    decel_width = float(t[right_idx] - t[left_idx])

    dx = float(x[idx_z_front] - x[idx_z_back])
    dy = float(y[idx_z_front] - y[idx_z_back])
    stride = float(np.hypot(dx, dy))
    stride_angle = float(np.degrees(np.arctan2(dx, dy))) if (dx != 0 or dy != 0) else 0.0

    return {
        "y_back": y_back,
        "y_back_score": y_back / w,
        "z_back": z_back,
        "z_back_score": z_back / w,
        "yz_back_score": (y_back + z_back) / w,
        "x_y_back": float(raw_fx[idx_y_back]),
        "y_front": y_front,
        "y_front_score": y_front / w,
        "z_front": z_front,
        "z_front_score": z_front / w,
        "yz_front_score": (y_front + z_front) / w,
        "x_y_front": float(raw_fx[idx_y_front]),
        "accel_impulse": accel_impulse,
        "accel_impulse_score": accel_impulse_score,
        "decel_impulse": decel_impulse,
        "decel_impulse_score": decel_impulse_score,
        "impulse_ratio": impulse_ratio,
        "z_transfer": float(t[idx_z_front] - t[idx_z_back]),
        "y_transfer": float(t[idx_y_front] - t[idx_y_back]),
        "yz_transfer_back": float(t[idx_z_back] - t[idx_y_back]),
        "yz_transfer_front": float(t[idx_z_front] - t[idx_y_front]),
        "clawback": clawback,
        "decel_width": decel_width,
        "stride": stride,
        "stride_angle": stride_angle,
    }


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_nf_features(ts: pd.DataFrame, weight_lb: float) -> Optional[dict]:
    """Compute aggregate NF features from a single pitch time series."""
    if ts is None or ts.empty or len(ts) < 5:
        return None

    ts = ts.sort_values("time_s").drop_duplicates(subset=["time_s"])
    if len(ts) < 5:
        return None

    t  = ts["time_s"].to_numpy(dtype=np.float64)
    raw_fx = ts["fx_lb"].to_numpy(dtype=np.float64)
    raw_fy = ts["fy_lb"].to_numpy(dtype=np.float64)
    raw_fz = ts["fz_lb"].to_numpy(dtype=np.float64)
    fx = raw_fx / max(weight_lb, 1e-6)
    fy = raw_fy / max(weight_lb, 1e-6)
    fz = raw_fz / max(weight_lb, 1e-6)
    x  = ts["x_in"].to_numpy(dtype=np.float64)
    y  = ts["y_in"].to_numpy(dtype=np.float64)

    if not (np.isfinite(fx).all() and np.isfinite(fy).all() and
            np.isfinite(fz).all() and np.isfinite(t).all()):
        return None

    # Resultant force magnitude at each timestep
    F = np.sqrt(fx**2 + fy**2 + fz**2)
    n = len(t)

    # Peak force
    peak_idx  = int(np.argmax(F))
    peak_F    = float(F[peak_idx])

    # Normalized time of peak (0–100)
    t_norm    = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    time_peak = float(t_norm[peak_idx]) * 100.0

    # Force impulse (area under force curve)
    force_impulse = float(np.trapezoid(F, t))

    # Late-phase force: final 20% of timesteps
    late_start = max(0, int(0.8 * n))
    force_at_late = float(np.mean(F[late_start:]))
    late_force_ratio = (force_at_late / peak_F * 100.0) if peak_F > 1e-9 else 0.0

    # Force buildup rate: from start to peak
    t_to_peak = t[peak_idx] - t[0]
    force_buildup_rate = (peak_F - float(F[0])) / max(t_to_peak, 1e-9)

    # Path movement speed and arc length
    if n > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        step_dist   = np.sqrt(dx**2 + dy**2)
        step_speed  = step_dist / np.maximum(dt, 1e-9)
        peak_path_speed = float(np.max(step_speed))
        path_arc_length = float(np.sum(step_dist))
    else:
        peak_path_speed = 0.0
        path_arc_length = 0.0

    features = {
        "peak_resultant_force": peak_F,
        "avg_resultant_force":  float(np.mean(F)),
        "force_impulse":        force_impulse,
        "peak_fx":              float(np.max(np.abs(fx))),
        "peak_fy":              float(np.max(np.abs(fy))),
        "peak_fz":              float(np.max(np.abs(fz))),
        "time_to_peak_force":   time_peak,
        "late_force_ratio":     late_force_ratio,
        "force_at_late":        force_at_late,
        "peak_path_speed":      peak_path_speed,
        "path_arc_length":      path_arc_length,
        "force_buildup_rate":   force_buildup_rate,
    }

    named_metrics = _compute_named_nf_metrics(t, raw_fx, raw_fy, raw_fz, x, y, weight_lb)
    if named_metrics is not None:
        features.update(named_metrics)

    return features


# ── BQ helpers ────────────────────────────────────────────────────────────────

def _fetch_scored_pitches(cfg: VelocityNFAnalysisConfig) -> pd.DataFrame:
    """Fetch all scored FB/SI/Cutter pitches with nf_file_name and velocity."""
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    scores_fq = _analysis(cfg, cfg.velocity_scores_table)
    sql = f"""
    SELECT
      player_full_name,
      TaggedPitchType,
      nf_file_name,
      SAFE_CAST(player_weight_lb AS FLOAT64) AS player_weight_lb,
      SAFE_CAST(pitch_velocity   AS FLOAT64) AS pitch_velocity
    FROM `{scores_fq}`
    WHERE dataset_id         = @dataset_id
      AND feature_version    = @feature_version
      AND TaggedPitchType IN UNNEST(@pitch_types)
      AND nf_file_name       IS NOT NULL
      AND pitch_velocity      IS NOT NULL
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
            bigquery.ArrayQueryParameter("pitch_types",      "STRING", list(HARD_PITCH_TYPES)),
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    df["pitch_velocity"]   = pd.to_numeric(df["pitch_velocity"],   errors="coerce")
    df["player_weight_lb"] = pd.to_numeric(df["player_weight_lb"], errors="coerce")
    return df.dropna(subset=["pitch_velocity", "player_weight_lb", "nf_file_name"])


def _fetch_time_series_batch(
    cfg: VelocityNFAnalysisConfig,
    client: bigquery.Client,
    file_names: List[str],
) -> pd.DataFrame:
    ts_fq = _ts_fq(cfg)
    sql = f"""
    SELECT file_name, time_s, fx_lb, fy_lb, fz_lb, x_in, y_in
    FROM `{ts_fq}`
    WHERE file_name IN UNNEST(@file_names)
    ORDER BY file_name, time_s
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("file_names", "STRING", file_names)]
    )
    return client.query(sql, job_config=job_config).to_dataframe()


# ── Correlation analysis ──────────────────────────────────────────────────────

def _build_coaching_text(
    feature: str,
    rank: int,
    r: float,
    mean_high: float,
    mean_low: float,
    n: int,
) -> str:
    meta  = FEATURE_META[feature]
    label = meta["label"]
    unit  = meta["unit"]
    diff  = mean_high - mean_low
    direction = "higher" if r > 0 else "lower"

    return (
        f"#{rank} velocity predictor: {label} (r={r:.2f}, n={n} pitches). "
        f"Hard throws average {mean_high:.2f} {unit} vs {mean_low:.2f} {unit} on soft throws "
        f"— a {abs(diff):.2f} {unit} difference. "
        f"{meta['coaching']}"
    )


def _analyze_player(
    player: str,
    pdf: pd.DataFrame,
    run_id: str,
    cfg: VelocityNFAnalysisConfig,
    created_at: datetime,
) -> List[dict]:
    """Run correlation analysis for a single player. Returns list of output rows."""
    velocities = pdf["pitch_velocity"].values
    n = len(pdf)

    if n < MIN_PITCHES:
        return []

    # Compute velocity percentile split
    p25 = float(np.percentile(velocities, 25))
    p75 = float(np.percentile(velocities, 75))

    rows = []
    rank_map: Dict[str, Tuple[float, float]] = {}  # feature → (|r|, r)

    for feat in FEATURE_NAMES:
        if feat not in pdf.columns:
            continue
        vals = pd.to_numeric(pdf[feat], errors="coerce")
        mask = vals.notna() & pd.to_numeric(pdf["pitch_velocity"], errors="coerce").notna()
        if mask.sum() < MIN_PITCHES:
            continue

        v_feat = vals[mask].values
        v_velo = pdf["pitch_velocity"][mask].values

        try:
            r, p_val = pearsonr(v_feat, v_velo)
        except Exception:
            continue

        if not np.isfinite(r):
            continue

        # High/low split means
        high_mask = v_velo >= p75
        low_mask  = v_velo <= p25
        mean_high = float(v_feat[high_mask].mean()) if high_mask.sum() >= 3 else float(v_feat.mean())
        mean_low  = float(v_feat[low_mask].mean())  if low_mask.sum()  >= 3 else float(v_feat.mean())

        rank_map[feat] = (abs(r), r, p_val, mean_high, mean_low, int(mask.sum()))

    if not rank_map:
        return []

    sorted_feats = sorted(rank_map.keys(), key=lambda f: rank_map[f][0], reverse=True)

    for rank_idx, feat in enumerate(sorted_feats, start=1):
        abs_r, r, p_val, mean_high, mean_low, n_feat = rank_map[feat]
        rows.append({
            "dataset_id":           cfg.dataset_id,
            "feature_version":      cfg.feature_version,
            "player_full_name":     player,
            "feature_name":         feat,
            "feature_label":        FEATURE_META[feat]["label"],
            "feature_category":     FEATURE_META[feat]["category"],
            "feature_unit":         FEATURE_META[feat]["unit"],
            "correlation":          round(r, 4),
            "abs_correlation":      round(abs_r, 4),
            "p_value":              round(p_val, 6),
            "rank":                 rank_idx,
            "n_pitches":            n_feat,
            "mean_feature_high_velo": round(mean_high, 4),
            "mean_feature_low_velo":  round(mean_low, 4),
            "coaching_text":        _build_coaching_text(
                feat, rank_idx, r, mean_high, mean_low, n_feat
            ),
            "run_id":               run_id,
            "created_at":           created_at,
        })

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def build_velocity_nf_analysis(cfg: VelocityNFAnalysisConfig) -> int:
    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    console.print(
        f"[bold]Velocity NF Analysis[/bold] "
        f"dataset={cfg.dataset_id} version={cfg.feature_version} "
        f"pitch_types={'+'.join(HARD_PITCH_TYPES)}"
    )
    console.print(f"Run: {run_id}")

    # 1. Fetch all scored pitches
    pitches = _fetch_scored_pitches(cfg)
    if pitches.empty:
        raise RuntimeError("No scored velocity pitches found.")

    n_players = pitches["player_full_name"].nunique()
    console.print(f"Pitches: {len(pitches):,} | players: {n_players}")

    # 2. Fetch NF time series in batches
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    file_names = pitches["nf_file_name"].unique().tolist()
    ts_parts: List[pd.DataFrame] = []

    for i in range(0, len(file_names), cfg.file_batch_size):
        batch = file_names[i : i + cfg.file_batch_size]
        part  = _fetch_time_series_batch(cfg, client, batch)
        if not part.empty:
            ts_parts.append(part)
        console.print(f"  TS batch {i//cfg.file_batch_size + 1}: {len(part):,} rows")

    if not ts_parts:
        raise RuntimeError("No NF time series data found.")

    ts_all    = pd.concat(ts_parts, ignore_index=True)
    ts_by_file = {fn: g for fn, g in ts_all.groupby("file_name")}
    console.print(f"Time series: {len(ts_all):,} rows | {len(ts_by_file):,} files")

    # 3. Compute per-pitch features
    feature_rows: List[dict] = []
    skipped = 0

    for _, r in pitches.iterrows():
        fn  = str(r["nf_file_name"])
        w   = float(r["player_weight_lb"])
        ts  = ts_by_file.get(fn)
        feats = _compute_nf_features(ts, w)
        if feats is None:
            skipped += 1
            continue
        feature_rows.append({
            "player_full_name": r["player_full_name"],
            "pitch_velocity":   float(r["pitch_velocity"]),
            **feats,
        })

    console.print(f"Feature rows: {len(feature_rows):,} | skipped: {skipped}")

    if not feature_rows:
        raise RuntimeError("No feature rows computed.")

    pdf_all = pd.DataFrame(feature_rows)

    # 4. Per-player correlation analysis
    output_rows: List[dict] = []
    players_analyzed = 0

    for player, pdf in pdf_all.groupby("player_full_name"):
        player_rows = _analyze_player(str(player), pdf.copy(), run_id, cfg, created_at)
        if player_rows:
            output_rows.extend(player_rows)
            players_analyzed += 1

    console.print(f"Players analyzed: {players_analyzed} | output rows: {len(output_rows):,}")

    if not output_rows:
        raise RuntimeError("No analysis rows produced.")

    # 5. Write to BQ
    df_out = pd.DataFrame(output_rows)
    dest_fq = _analysis(cfg, cfg.output_table)

    job_config = bigquery.QueryJobConfig()
    load_cfg = bigquery.LoadJobConfig(write_disposition=cfg.write_disposition)
    job = client.load_table_from_dataframe(df_out, dest_fq, job_config=load_cfg)
    job.result()

    n_rows = client.get_table(dest_fq).num_rows
    console.print(f"[green]✅ {cfg.output_table}[/green] rows={n_rows:,}")
    return n_rows


def load_cfg_from_env(dataset_id: str, feature_version: str) -> VelocityNFAnalysisConfig:
    return VelocityNFAnalysisConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        bq_src_dataset=(os.getenv("BQ_SRC_DATASET") or dataset_id).strip(),
        velocity_scores_table=(os.getenv("BQ_VELOCITY_SCORES_TABLE") or "scores_model_velocity").strip(),
        nf_time_series_table=(os.getenv("BQ_NF_TIME_SERIES_TABLE") or "gold_newtforce_time_series").strip(),
        output_table=(os.getenv("BQ_VELOCITY_NF_ANALYSIS_TABLE") or "velocity_nf_analysis_v1").strip(),
        file_batch_size=int(os.getenv("NF_TS_FILE_BATCH_SIZE", "600")),
    )
