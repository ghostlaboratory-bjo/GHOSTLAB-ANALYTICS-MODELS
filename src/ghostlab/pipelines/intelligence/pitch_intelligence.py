from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

from google.cloud import bigquery
from rich.console import Console

from ghostlab.io.bq import BQConfig, bq_client

console = Console()


@dataclass(frozen=True)
class IntelBuildConfig:
    dataset_id: str
    feature_version: str

    gcp_project: str = ""
    bq_location: str = "us-east5"

    # Source datasets
    bq_analysis_dataset: str = "analysis_model"
    bq_src_dataset: str = ""  # where pitch_core and NF time series live

    # Source tables
    pitch_core_table: str = "pitch_core_v1"
    nf_time_series_table: str = "gold_newtforce_time_series"
    command_scores_table: str = "model_command_scores_v1"
    miss_direction_table: str = "model_miss_direction_v1"

    # Output tables
    intel_table: str = "model_pitch_intelligence_v1"
    mechanics_table: str = "mechanics_detail_v1"
    session_trend_table: str = "session_trend_v1"
    summary_table: str = "coach_intelligence_summary_v1"

    write_disposition: str = "WRITE_TRUNCATE"


# ── fully-qualified name helpers ──────────────────────────────────────────────

def _fq(cfg: IntelBuildConfig, dataset: str, table: str) -> str:
    return f"{cfg.gcp_project}.{dataset}.{table}"

def _analysis(cfg: IntelBuildConfig, table: str) -> str:
    return _fq(cfg, cfg.bq_analysis_dataset, table)

def _src(cfg: IntelBuildConfig, table: str) -> str:
    return _fq(cfg, cfg.bq_src_dataset or cfg.dataset_id, table)


# ── shared BQ write helper ────────────────────────────────────────────────────

def _run_sql_to_table(
    client: bigquery.Client,
    sql: str,
    dest_fq: str,
    write_disposition: str,
    params: list | None = None,
    location: str = "us-east5",
) -> int:
    job_config = bigquery.QueryJobConfig(
        destination=dest_fq,
        write_disposition=write_disposition,
        create_disposition="CREATE_IF_NEEDED",
    )
    if params:
        job_config.query_parameters = params
    job = client.query(sql, job_config=job_config, location=location)
    job.result()
    return client.get_table(dest_fq).num_rows


# ── Step 1: model_pitch_intelligence_v1 ───────────────────────────────────────

def _build_intel_table(cfg: IntelBuildConfig, client: bigquery.Client) -> int:
    console.print("[bold]Step 1:[/bold] Building model_pitch_intelligence_v1")

    command_fq = _analysis(cfg, cfg.command_scores_table)
    miss_fq    = _analysis(cfg, cfg.miss_direction_table)
    dest_fq    = _analysis(cfg, cfg.intel_table)

    sql = f"""
    SELECT
      c.dataset_id,
      c.feature_version,
      c.team_id,
      c.PitchID,
      c.nf_file_name,
      c.event_ts,
      c.session_date,
      c.player_full_name,
      c.PitcherId,
      c.PitcherThrows,
      c.TaggedPitchType,

      ROUND(c.command_score, 2)                         AS command_score,
      CASE
        WHEN c.command_score >= 90 THEN 'Elite'
        WHEN c.command_score >= 80 THEN 'Plus'
        WHEN c.command_score >= 70 THEN 'Average'
        WHEN c.command_score >= 60 THEN 'Below Avg'
        ELSE 'Poor'
      END                                               AS command_grade,
      CONCAT(
        CASE
          WHEN c.command_score >= 90 THEN 'Elite'
          WHEN c.command_score >= 80 THEN 'Plus'
          WHEN c.command_score >= 70 THEN 'Average'
          WHEN c.command_score >= 60 THEN 'Below Avg'
          ELSE 'Poor'
        END,
        ' command (',
        CAST(CAST(ROUND(c.command_score) AS INT64) AS STRING),
        ')'
      )                                                 AS command_summary,

      c.actual_miss_distance_in,
      c.predicted_miss_distance_in,

      m.predicted_horizontal_miss_type,
      m.predicted_vertical_miss_type,
      CASE
        WHEN m.predicted_horizontal_miss_type = 'In-Zone'
         AND m.predicted_vertical_miss_type   = 'In-Zone' THEN 'In-Zone'
        WHEN m.predicted_horizontal_miss_type = 'In-Zone' THEN m.predicted_vertical_miss_type
        WHEN m.predicted_vertical_miss_type   = 'In-Zone' THEN m.predicted_horizontal_miss_type
        ELSE CONCAT(m.predicted_horizontal_miss_type, ' / ', m.predicted_vertical_miss_type)
      END                                               AS predicted_miss_combined,
      ROUND(m.horizontal_confidence, 4)                AS horizontal_confidence,
      ROUND(m.vertical_confidence, 4)                  AS vertical_confidence,

      c.plate_x_ft,
      c.plate_z_ft,
      c.is_strike_target,
      c.arm_side_miss_ft,
      c.outside_x_in,
      c.outside_z_in,

      CURRENT_TIMESTAMP()                               AS created_at

    FROM `{command_fq}` c
    JOIN `{miss_fq}` m
      ON  c.PitchID          = m.PitchID
      AND c.dataset_id       = m.dataset_id
      AND c.feature_version  = m.feature_version
    WHERE c.dataset_id      = @dataset_id
      AND c.feature_version = @feature_version
    """

    params = [
        bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
        bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
    ]

    rows = _run_sql_to_table(client, sql, dest_fq, cfg.write_disposition, params, cfg.bq_location)
    console.print(f"[green]✅ {cfg.intel_table}[/green] rows={rows:,}")
    return rows


# ── Step 2: mechanics_detail_v1 ───────────────────────────────────────────────

def _build_mechanics_table(cfg: IntelBuildConfig, client: bigquery.Client) -> int:
    console.print("[bold]Step 2:[/bold] Building mechanics_detail_v1")

    intel_fq      = _analysis(cfg, cfg.intel_table)
    pitch_core_fq = _analysis(cfg, cfg.pitch_core_table)
    nf_ts_fq      = _src(cfg, cfg.nf_time_series_table)
    dest_fq       = _analysis(cfg, cfg.mechanics_table)

    sql = f"""
    WITH nf_agg AS (
      SELECT
        file_name,
        COUNT(*)                                           AS nf_points,
        MAX(ABS(SAFE_CAST(fx_lb AS FLOAT64)))             AS peak_fx_lb,
        MAX(ABS(SAFE_CAST(fy_lb AS FLOAT64)))             AS peak_fy_lb,
        MAX(ABS(SAFE_CAST(fz_lb AS FLOAT64)))             AS peak_fz_lb,
        AVG(ABS(SAFE_CAST(fx_lb AS FLOAT64)))             AS avg_abs_fx_lb,
        AVG(ABS(SAFE_CAST(fy_lb AS FLOAT64)))             AS avg_abs_fy_lb,
        AVG(ABS(SAFE_CAST(fz_lb AS FLOAT64)))             AS avg_abs_fz_lb,
        STDDEV(SAFE_CAST(fx_lb AS FLOAT64))               AS fx_std,
        STDDEV(SAFE_CAST(fy_lb AS FLOAT64))               AS fy_std,
        STDDEV(SAFE_CAST(fz_lb AS FLOAT64))               AS fz_std,
        STDDEV(SAFE_CAST(x_in AS FLOAT64))                AS path_x_std,
        STDDEV(SAFE_CAST(y_in AS FLOAT64))                AS path_y_std
      FROM `{nf_ts_fq}`
      GROUP BY file_name
    )

    SELECT
      i.dataset_id,
      i.feature_version,
      i.team_id,
      i.PitchID,
      i.player_full_name,
      i.PitcherId,
      i.PitcherThrows,
      i.TaggedPitchType,
      i.session_date,
      i.event_ts,

      i.command_score,
      i.command_grade,
      i.predicted_miss_combined,
      i.predicted_horizontal_miss_type,
      i.predicted_vertical_miss_type,
      i.horizontal_confidence,
      i.vertical_confidence,

      p.nf_file_name,
      SAFE_CAST(p.RelHeight       AS FLOAT64)   AS RelHeight,
      SAFE_CAST(p.RelSide         AS FLOAT64)   AS RelSide,
      SAFE_CAST(p.Extension       AS FLOAT64)   AS Extension,
      SAFE_CAST(p.VertRelAngle    AS FLOAT64)   AS VertRelAngle,
      SAFE_CAST(p.HorzRelAngle    AS FLOAT64)   AS HorzRelAngle,
      SAFE_CAST(p.pitch_velocity  AS FLOAT64)   AS RelSpeed,
      SAFE_CAST(p.InducedVertBreak AS FLOAT64)  AS InducedVertBreak,
      SAFE_CAST(p.HorzBreak       AS FLOAT64)   AS HorzBreak,

      COALESCE(n.nf_points, 0)                  AS nf_points,
      n.peak_fx_lb,
      n.peak_fy_lb,
      n.peak_fz_lb,
      n.avg_abs_fx_lb,
      n.avg_abs_fy_lb,
      n.avg_abs_fz_lb,
      n.fx_std,
      n.fy_std,
      n.fz_std,
      n.path_x_std,
      n.path_y_std,

      CURRENT_TIMESTAMP()                        AS created_at

    FROM `{intel_fq}` i
    JOIN `{pitch_core_fq}` p
      ON  i.PitchID         = p.PitchID
      AND i.dataset_id      = p.dataset_id
      AND i.feature_version = p.feature_version
    LEFT JOIN nf_agg n
      ON p.nf_file_name = n.file_name
    WHERE i.dataset_id      = @dataset_id
      AND i.feature_version = @feature_version
    """

    params = [
        bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
        bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
    ]

    rows = _run_sql_to_table(client, sql, dest_fq, cfg.write_disposition, params, cfg.bq_location)
    console.print(f"[green]✅ {cfg.mechanics_table}[/green] rows={rows:,}")
    return rows


# ── Step 3: session_trend_v1 ──────────────────────────────────────────────────

def _build_session_trend_table(cfg: IntelBuildConfig, client: bigquery.Client) -> int:
    console.print("[bold]Step 3:[/bold] Building session_trend_v1")

    intel_fq = _analysis(cfg, cfg.intel_table)
    dest_fq  = _analysis(cfg, cfg.session_trend_table)

    sql = f"""
    WITH session_base AS (
      SELECT
        dataset_id,
        feature_version,
        team_id,
        player_full_name,
        session_date,
        COUNT(*)                                                               AS pitches,
        ROUND(AVG(command_score), 2)                                          AS command_score,
        APPROX_TOP_COUNT(predicted_miss_combined, 1)[OFFSET(0)].value        AS main_miss_pattern,
        APPROX_TOP_COUNT(TaggedPitchType, 1)[OFFSET(0)].value                AS main_pitch_type,
        ROUND(AVG(horizontal_confidence), 4)                                  AS avg_horizontal_confidence,
        ROUND(AVG(vertical_confidence), 4)                                    AS avg_vertical_confidence
      FROM `{intel_fq}`
      WHERE dataset_id      = @dataset_id
        AND feature_version = @feature_version
      GROUP BY 1, 2, 3, 4, 5
    ),

    with_window AS (
      SELECT
        *,
        LAG(command_score) OVER (
          PARTITION BY dataset_id, feature_version, player_full_name
          ORDER BY session_date
        ) AS previous_command_score,
        ROUND(
          AVG(command_score) OVER (
            PARTITION BY dataset_id, feature_version, player_full_name
            ORDER BY session_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
          ), 2
        ) AS command_score_3_session_avg
      FROM session_base
    )

    SELECT
      dataset_id,
      feature_version,
      team_id,
      player_full_name,
      session_date,
      pitches,
      command_score,
      previous_command_score,
      ROUND(command_score - COALESCE(previous_command_score, command_score), 2) AS command_score_change,
      command_score_3_session_avg,
      CASE
        WHEN command_score - COALESCE(previous_command_score, command_score) >=  5 THEN 'Improving'
        WHEN command_score - COALESCE(previous_command_score, command_score) <= -5 THEN 'Declining'
        ELSE 'Stable'
      END                           AS trend_label,
      main_miss_pattern,
      main_pitch_type,
      avg_horizontal_confidence,
      avg_vertical_confidence,
      CURRENT_TIMESTAMP()           AS created_at
    FROM with_window
    """

    params = [
        bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
        bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
    ]

    rows = _run_sql_to_table(client, sql, dest_fq, cfg.write_disposition, params, cfg.bq_location)
    console.print(f"[green]✅ {cfg.session_trend_table}[/green] rows={rows:,}")
    return rows


# ── Step 4: coach_intelligence_summary_v1 ────────────────────────────────────

def _build_summary_table(cfg: IntelBuildConfig, client: bigquery.Client) -> int:
    console.print("[bold]Step 4:[/bold] Building coach_intelligence_summary_v1")

    mechanics_fq     = _analysis(cfg, cfg.mechanics_table)
    session_trend_fq = _analysis(cfg, cfg.session_trend_table)
    dest_fq          = _analysis(cfg, cfg.summary_table)

    sql = f"""
    WITH mech_split AS (
      -- Compare avg mechanical values between high-command and low-command pitches per player
      SELECT
        dataset_id,
        feature_version,
        player_full_name,

        AVG(IF(command_score >= 80, RelSide,   NULL)) AS high_rel_side,
        AVG(IF(command_score >= 80, RelHeight,  NULL)) AS high_rel_height,
        AVG(IF(command_score >= 80, Extension,  NULL)) AS high_extension,
        AVG(IF(command_score >= 80, fx_std,     NULL)) AS high_fx_std,
        AVG(IF(command_score >= 80, fy_std,     NULL)) AS high_fy_std,
        AVG(IF(command_score >= 80, fz_std,     NULL)) AS high_fz_std,
        AVG(IF(command_score >= 80, path_x_std, NULL)) AS high_path_x_std,
        AVG(IF(command_score >= 80, path_y_std, NULL)) AS high_path_y_std,

        AVG(IF(command_score < 70, RelSide,   NULL)) AS low_rel_side,
        AVG(IF(command_score < 70, RelHeight,  NULL)) AS low_rel_height,
        AVG(IF(command_score < 70, Extension,  NULL)) AS low_extension,
        AVG(IF(command_score < 70, fx_std,     NULL)) AS low_fx_std,
        AVG(IF(command_score < 70, fy_std,     NULL)) AS low_fy_std,
        AVG(IF(command_score < 70, fz_std,     NULL)) AS low_fz_std,
        AVG(IF(command_score < 70, path_x_std, NULL)) AS low_path_x_std,
        AVG(IF(command_score < 70, path_y_std, NULL)) AS low_path_y_std,

        COUNT(IF(command_score >= 80, 1, NULL)) AS high_count,
        COUNT(IF(command_score <  70, 1, NULL)) AS low_count
      FROM `{mechanics_fq}`
      WHERE dataset_id      = @dataset_id
        AND feature_version = @feature_version
      GROUP BY 1, 2, 3
    ),

    drivers AS (
      -- Compute absolute delta for each mechanical dimension
      SELECT
        dataset_id,
        feature_version,
        player_full_name,
        ABS(COALESCE(low_rel_side,   0) - COALESCE(high_rel_side,   0)) AS rel_side_diff,
        ABS(COALESCE(low_rel_height, 0) - COALESCE(high_rel_height, 0)) AS rel_height_diff,
        ABS(COALESCE(low_extension,  0) - COALESCE(high_extension,  0)) AS extension_diff,
        ABS(
          (COALESCE(low_fx_std,  0) + COALESCE(low_fy_std,  0) + COALESCE(low_fz_std,  0)) -
          (COALESCE(high_fx_std, 0) + COALESCE(high_fy_std, 0) + COALESCE(high_fz_std, 0))
        ) / 10.0 AS force_diff,
        ABS(
          (COALESCE(low_path_x_std,  0) + COALESCE(low_path_y_std,  0)) -
          (COALESCE(high_path_x_std, 0) + COALESCE(high_path_y_std, 0))
        ) / 10.0 AS path_diff,
        high_count,
        low_count
      FROM mech_split
    ),

    session_agg AS (
      -- Roll up session trend into a single per-player snapshot
      SELECT
        dataset_id,
        feature_version,
        team_id,
        player_full_name,
        ROUND(AVG(command_score), 2)                                   AS command_score,
        SUM(pitches)                                                   AS pitch_count,
        APPROX_TOP_COUNT(main_miss_pattern, 1)[OFFSET(0)].value       AS main_miss_pattern,
        MAX(created_at)                                                AS created_at
      FROM `{session_trend_fq}`
      WHERE dataset_id      = @dataset_id
        AND feature_version = @feature_version
      GROUP BY 1, 2, 3, 4
    )

    SELECT
      sa.dataset_id,
      sa.feature_version,
      sa.team_id,
      sa.player_full_name,
      sa.command_score,
      CASE
        WHEN sa.command_score >= 90 THEN 'Elite'
        WHEN sa.command_score >= 80 THEN 'Plus'
        WHEN sa.command_score >= 70 THEN 'Average'
        WHEN sa.command_score >= 60 THEN 'Below Avg'
        ELSE 'Poor'
      END                                               AS command_grade,
      sa.pitch_count,
      sa.main_miss_pattern,

      -- Primary driver: the mechanical dimension with largest high-vs-low delta
      CASE
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_side_diff   THEN 'Release Side Shift'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_height_diff THEN 'Release Height Shift'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.extension_diff  THEN 'Extension Shift'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.force_diff      THEN 'NewtForce Force Stability'
        ELSE                          'NewtForce Path Stability'
      END                                               AS primary_driver,

      -- Teaching priority: crosses primary driver with dominant miss pattern
      CASE
        WHEN sa.command_score < 60
          THEN 'Overall Command Stability'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_side_diff AND sa.main_miss_pattern LIKE '%Arm-Side%'
          THEN 'Release Point — Arm Side'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_side_diff AND sa.main_miss_pattern LIKE '%Glove-Side%'
          THEN 'Release Point — Glove Side'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_side_diff
          THEN 'Release Point Consistency'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_height_diff AND sa.main_miss_pattern LIKE '%High%'
          THEN 'Extension and Finish'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_height_diff AND sa.main_miss_pattern LIKE '%Low%'
          THEN 'Release Height Consistency'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.rel_height_diff
          THEN 'Release Height Consistency'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.extension_diff
          THEN 'Timing — Extension Consistency'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.force_diff
          THEN 'Force Production Consistency'
        WHEN GREATEST(d.rel_side_diff, d.rel_height_diff, d.extension_diff, d.force_diff, d.path_diff)
             = d.path_diff
          THEN 'Delivery Path Repeatability'
        ELSE 'Release Consistency'
      END                                               AS teaching_priority,

      -- best_pattern fields populated downstream when best-pattern pipeline runs
      CAST(NULL AS FLOAT64)                             AS best_pattern_similarity,
      CAST(NULL AS STRING)                              AS best_pattern_status,

      CURRENT_TIMESTAMP()                               AS created_at

    FROM session_agg sa
    LEFT JOIN drivers d
      ON  sa.dataset_id       = d.dataset_id
      AND sa.feature_version  = d.feature_version
      AND sa.player_full_name = d.player_full_name
    """

    params = [
        bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
        bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
    ]

    rows = _run_sql_to_table(client, sql, dest_fq, cfg.write_disposition, params, cfg.bq_location)
    console.print(f"[green]✅ {cfg.summary_table}[/green] rows={rows:,}")
    return rows


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_pitch_intelligence(cfg: IntelBuildConfig) -> Dict[str, int]:
    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    console.print(f"[bold]Intel Build[/bold] dataset={cfg.dataset_id} version={cfg.feature_version}")
    console.print(f"Run: {run_id}")

    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))

    intel_rows    = _build_intel_table(cfg, client)
    mech_rows     = _build_mechanics_table(cfg, client)
    trend_rows    = _build_session_trend_table(cfg, client)
    summary_rows  = _build_summary_table(cfg, client)

    console.print(f"[bold green]DONE[/bold green] run_id={run_id}")

    return {
        "intel_rows":   intel_rows,
        "mech_rows":    mech_rows,
        "trend_rows":   trend_rows,
        "summary_rows": summary_rows,
        "run_id":       run_id,
    }


def load_cfg_from_env(dataset_id: str, feature_version: str) -> IntelBuildConfig:
    return IntelBuildConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        bq_src_dataset=(os.getenv("BQ_SRC_DATASET") or dataset_id).strip(),
        pitch_core_table=(os.getenv("BQ_PITCH_CORE_TABLE") or "pitch_core_v1").strip(),
        nf_time_series_table=(os.getenv("BQ_NF_TIME_SERIES_TABLE") or "gold_newtforce_time_series").strip(),
        command_scores_table=(os.getenv("BQ_COMMAND_MODEL_TABLE") or "model_command_scores_v1").strip(),
        miss_direction_table=(os.getenv("BQ_MISS_DIRECTION_MODEL_TABLE") or "model_miss_direction_v1").strip(),
        intel_table=(os.getenv("BQ_INTEL_TABLE") or "model_pitch_intelligence_v1").strip(),
        mechanics_table=(os.getenv("BQ_MECHANICS_TABLE") or "mechanics_detail_v1").strip(),
        session_trend_table=(os.getenv("BQ_SESSION_TREND_TABLE") or "session_trend_v1").strip(),
        summary_table=(os.getenv("BQ_SUMMARY_TABLE") or "coach_intelligence_summary_v1").strip(),
        write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
    )
