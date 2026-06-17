# src/ghostlab/pipelines/intelligence/velocity_intelligence.py
"""
Velocity Intelligence Build — completely separate from the accuracy intel pipeline.

Reads raw per-pitch velocity scores (scores_model_velocity) and the player
baseline meta (baseline_model_velocity) and produces two pre-aggregated tables:

  Step 1 → velocity_session_trend_v1
    One row per player / session.
    Shows: avg velocity, velocity vs baseline, pattern similarity, trend labels.

  Step 2 → velocity_coaching_summary_v1
    One row per player (lifetime summary).
    The key coaching insight: avg velocity when mechanics match the max-effort
    pattern (sim >= 75%) vs when they drift (sim < 60%), with the computed
    velocity gap and a human-readable coaching_insight text field.
"""
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
class VelocityIntelConfig:
    dataset_id: str
    feature_version: str

    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_analysis_dataset: str = "analysis_model"

    # Source tables (all in bq_analysis_dataset)
    velocity_scores_table: str = "scores_model_velocity"
    velocity_baseline_table: str = "baseline_model_velocity"

    # Output tables
    session_trend_table: str = "velocity_session_trend_v1"
    coaching_summary_table: str = "velocity_coaching_summary_v1"

    write_disposition: str = "WRITE_TRUNCATE"


# ── helpers ───────────────────────────────────────────────────────────────────

def _analysis(cfg: VelocityIntelConfig, table: str) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{table}"


def _run_sql_to_table(
    client: bigquery.Client,
    sql: str,
    dest_fq: str,
    write_disposition: str,
    params: list,
    location: str,
) -> int:
    job_config = bigquery.QueryJobConfig(
        destination=dest_fq,
        write_disposition=write_disposition,
        create_disposition="CREATE_IF_NEEDED",
        query_parameters=params,
    )
    job = client.query(sql, job_config=job_config, location=location)
    job.result()
    return client.get_table(dest_fq).num_rows


# ── Step 1: velocity_session_trend_v1 ────────────────────────────────────────

def _build_session_trend(cfg: VelocityIntelConfig, client: bigquery.Client) -> int:
    console.print("[bold]Step 1:[/bold] Building velocity_session_trend_v1")

    scores_fq = _analysis(cfg, cfg.velocity_scores_table)
    dest_fq   = _analysis(cfg, cfg.session_trend_table)

    sql = f"""
    WITH session_base AS (
      SELECT
        dataset_id,
        feature_version,
        player_full_name,
        session_date,
        COUNT(*)                                                            AS n_pitches,
        ROUND(AVG(pitch_velocity), 1)                                      AS avg_velocity,
        ROUND(AVG(velocity_vs_baseline), 2)                               AS avg_velocity_vs_baseline,
        ROUND(AVG(pattern_similarity), 1)                                  AS avg_pattern_similarity,
        ROUND(MAX(pitch_velocity), 1)                                      AS best_velocity,
        ROUND(100.0 * COUNTIF(velocity_vs_baseline > 0) / COUNT(*), 1)   AS pct_above_baseline,
        ROUND(AVG(similarity_score), 1)                                    AS avg_nf_similarity,
        ANY_VALUE(baseline_mean_velocity)                                  AS baseline_mean_velocity
      FROM `{scores_fq}`
      WHERE dataset_id      = @dataset_id
        AND feature_version = @feature_version
        AND pitch_velocity      IS NOT NULL
        AND velocity_vs_baseline IS NOT NULL
      GROUP BY 1, 2, 3, 4
    ),

    with_window AS (
      SELECT
        *,
        LAG(avg_velocity) OVER (
          PARTITION BY dataset_id, feature_version, player_full_name
          ORDER BY session_date
        ) AS previous_avg_velocity,
        ROUND(
          AVG(avg_velocity) OVER (
            PARTITION BY dataset_id, feature_version, player_full_name
            ORDER BY session_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
          ), 1
        ) AS velocity_3_session_avg,
        LAG(avg_pattern_similarity) OVER (
          PARTITION BY dataset_id, feature_version, player_full_name
          ORDER BY session_date
        ) AS previous_avg_pattern_similarity
      FROM session_base
    )

    SELECT
      dataset_id,
      feature_version,
      player_full_name,
      session_date,
      n_pitches,
      avg_velocity,
      avg_velocity_vs_baseline,
      avg_pattern_similarity,
      best_velocity,
      pct_above_baseline,
      avg_nf_similarity,
      baseline_mean_velocity,
      previous_avg_velocity,
      ROUND(avg_velocity - COALESCE(previous_avg_velocity, avg_velocity), 2) AS velocity_change,
      CASE
        WHEN avg_velocity - COALESCE(previous_avg_velocity, avg_velocity) >=  0.5 THEN 'Improving'
        WHEN avg_velocity - COALESCE(previous_avg_velocity, avg_velocity) <= -0.5 THEN 'Declining'
        ELSE 'Stable'
      END                                                                  AS trend_label,
      velocity_3_session_avg,
      previous_avg_pattern_similarity,
      ROUND(
        avg_pattern_similarity - COALESCE(previous_avg_pattern_similarity, avg_pattern_similarity), 1
      )                                                                    AS pattern_similarity_change,
      CURRENT_TIMESTAMP()                                                  AS created_at
    FROM with_window
    ORDER BY player_full_name, session_date
    """

    params = [
        bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
        bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
    ]

    rows = _run_sql_to_table(client, sql, dest_fq, cfg.write_disposition, params, cfg.bq_location)
    console.print(f"[green]✅ {cfg.session_trend_table}[/green] rows={rows:,}")
    return rows


# ── Step 2: velocity_coaching_summary_v1 ──────────────────────────────────────

def _build_coaching_summary(cfg: VelocityIntelConfig, client: bigquery.Client) -> int:
    console.print("[bold]Step 2:[/bold] Building velocity_coaching_summary_v1")

    scores_fq   = _analysis(cfg, cfg.velocity_scores_table)
    baseline_fq = _analysis(cfg, cfg.velocity_baseline_table)
    dest_fq     = _analysis(cfg, cfg.coaching_summary_table)

    sql = f"""
    WITH

    -- Baseline meta: one row per player (the max-effort pattern signature)
    baseline_meta AS (
      SELECT
        player_full_name,
        MAX(mean_velocity)    AS baseline_mean_velocity,
        MAX(stddev_velocity)  AS stddev_velocity,
        MAX(n_pitches_used)   AS n_baseline_pitches
      FROM `{baseline_fq}`
      WHERE dataset_id      = @dataset_id
        AND feature_version = @feature_version
      GROUP BY player_full_name
    ),

    -- All pitch-level stats
    overall_stats AS (
      SELECT
        player_full_name,
        ROUND(AVG(pitch_velocity),        1)  AS avg_velocity,
        ROUND(AVG(pattern_similarity),    1)  AS avg_pattern_similarity,
        ROUND(AVG(velocity_vs_baseline),  2)  AS avg_velocity_vs_baseline,
        ROUND(MAX(pitch_velocity),        1)  AS peak_velocity,
        -- velocity when mechanics are on (sim >= 75)
        ROUND(AVG(IF(pattern_similarity >= 75, pitch_velocity, NULL)), 1) AS avg_velo_when_sim_high,
        -- velocity when mechanics drift (sim < 60)
        ROUND(AVG(IF(pattern_similarity <  60, pitch_velocity, NULL)), 1) AS avg_velo_when_sim_low,
        COUNTIF(pattern_similarity >= 75)                                 AS n_high_sim_pitches,
        COUNTIF(pattern_similarity <  60)                                 AS n_low_sim_pitches,
        ROUND(100.0 * COUNTIF(pattern_similarity >= 75) / COUNT(*), 1)   AS pct_high_sim
      FROM `{scores_fq}`
      WHERE dataset_id         = @dataset_id
        AND feature_version    = @feature_version
        AND pitch_velocity      IS NOT NULL
        AND velocity_vs_baseline IS NOT NULL
      GROUP BY player_full_name
    ),

    -- Session-level counts
    session_counts AS (
      SELECT
        player_full_name,
        COUNT(DISTINCT session_date)  AS n_sessions,
        MAX(session_date)             AS last_session_date,
        COUNT(*)                      AS total_pitches
      FROM `{scores_fq}`
      WHERE dataset_id      = @dataset_id
        AND feature_version = @feature_version
        AND pitch_velocity IS NOT NULL
      GROUP BY player_full_name
    ),

    -- Last session stats
    last_session_stats AS (
      SELECT
        s.player_full_name,
        ROUND(AVG(s.pitch_velocity),     1)  AS last_session_avg_velocity,
        ROUND(AVG(s.pattern_similarity), 1)  AS last_session_avg_pattern_similarity,
        COUNT(*)                             AS last_session_n_pitches
      FROM `{scores_fq}` s
      INNER JOIN (
        SELECT player_full_name, MAX(session_date) AS max_date
        FROM `{scores_fq}`
        WHERE dataset_id      = @dataset_id
          AND feature_version = @feature_version
          AND pitch_velocity IS NOT NULL
        GROUP BY player_full_name
      ) mx
        ON s.player_full_name = mx.player_full_name
       AND s.session_date     = mx.max_date
      WHERE s.dataset_id      = @dataset_id
        AND s.feature_version = @feature_version
        AND s.pitch_velocity IS NOT NULL
      GROUP BY s.player_full_name
    ),

    -- Trend: last 3 sessions avg velocity vs prior 3 sessions
    session_ranked AS (
      SELECT
        player_full_name,
        session_date,
        AVG(pitch_velocity) AS session_avg_velocity,
        ROW_NUMBER() OVER (PARTITION BY player_full_name ORDER BY session_date DESC) AS sess_rank
      FROM `{scores_fq}`
      WHERE dataset_id      = @dataset_id
        AND feature_version = @feature_version
        AND pitch_velocity IS NOT NULL
      GROUP BY player_full_name, session_date
    ),

    trend_calc AS (
      SELECT
        player_full_name,
        ROUND(AVG(IF(sess_rank <= 3, session_avg_velocity, NULL)), 1)          AS last3_avg,
        ROUND(AVG(IF(sess_rank BETWEEN 4 AND 6, session_avg_velocity, NULL)), 1) AS prior3_avg
      FROM session_ranked
      GROUP BY player_full_name
    ),

    -- Velocity ceiling: avg of each player's personal top 10% pitches by velocity
    velocity_ceiling AS (
      SELECT
        player_full_name,
        ROUND(AVG(pitch_velocity), 1) AS velocity_ceiling
      FROM (
        SELECT
          player_full_name,
          pitch_velocity,
          NTILE(10) OVER (
            PARTITION BY player_full_name ORDER BY pitch_velocity DESC
          ) AS velo_decile
        FROM `{scores_fq}`
        WHERE dataset_id      = @dataset_id
          AND feature_version = @feature_version
          AND pitch_velocity IS NOT NULL
      )
      WHERE velo_decile = 1
      GROUP BY player_full_name
    )

    SELECT
      @dataset_id      AS dataset_id,
      @feature_version AS feature_version,

      bm.player_full_name,

      -- Baseline signature
      bm.baseline_mean_velocity,
      bm.stddev_velocity,
      bm.n_baseline_pitches,

      -- Overall pitch-level stats
      os.avg_velocity,
      os.avg_pattern_similarity,
      os.avg_velocity_vs_baseline,
      os.peak_velocity,
      vc.velocity_ceiling,

      -- The money columns: velo split by mechanics quality
      os.avg_velo_when_sim_high,
      os.avg_velo_when_sim_low,
      ROUND(
        COALESCE(os.avg_velo_when_sim_high, 0) - COALESCE(os.avg_velo_when_sim_low, 0), 1
      )                                          AS velo_gap,
      os.n_high_sim_pitches,
      os.n_low_sim_pitches,
      os.pct_high_sim,

      -- Session meta
      sc.n_sessions,
      sc.last_session_date,
      sc.total_pitches,

      -- Last session snapshot
      ls.last_session_avg_velocity,
      ls.last_session_avg_pattern_similarity,
      ls.last_session_n_pitches,

      -- Trend direction (last 3 sessions vs prior 3)
      CASE
        WHEN tc.last3_avg IS NOT NULL AND tc.prior3_avg IS NOT NULL
          AND tc.last3_avg - tc.prior3_avg >=  0.5 THEN 'Improving'
        WHEN tc.last3_avg IS NOT NULL AND tc.prior3_avg IS NOT NULL
          AND tc.last3_avg - tc.prior3_avg <= -0.5 THEN 'Declining'
        ELSE 'Stable'
      END                                        AS trend_direction,
      tc.last3_avg                               AS trend_last3_avg,
      tc.prior3_avg                              AS trend_prior3_avg,

      -- Human-readable coaching insight surfaced directly on the website
      CASE
        WHEN os.avg_velo_when_sim_high IS NOT NULL
          AND os.avg_velo_when_sim_low IS NOT NULL
          AND ROUND(os.avg_velo_when_sim_high - os.avg_velo_when_sim_low, 1) > 0
        THEN CONCAT(
          'When mechanics match your max-effort pattern (≥75% match), avg velocity is ',
          CAST(os.avg_velo_when_sim_high AS STRING), ' mph — ',
          CAST(ROUND(os.avg_velo_when_sim_high - os.avg_velo_when_sim_low, 1) AS STRING),
          ' mph above when mechanics drift (<60% match).'
        )
        WHEN os.avg_velo_when_sim_high IS NOT NULL
        THEN CONCAT(
          'Max-effort mechanics average ', CAST(os.avg_velo_when_sim_high AS STRING),
          ' mph. Keep tracking to build a full velocity-mechanics picture.'
        )
        ELSE 'Not enough data yet to compute a velocity coaching insight.'
      END                                        AS coaching_insight,

      CURRENT_TIMESTAMP()                        AS created_at

    FROM baseline_meta bm
    JOIN  overall_stats     os ON bm.player_full_name = os.player_full_name
    JOIN  session_counts    sc ON bm.player_full_name = sc.player_full_name
    LEFT JOIN last_session_stats ls ON bm.player_full_name = ls.player_full_name
    LEFT JOIN trend_calc    tc ON bm.player_full_name = tc.player_full_name
    LEFT JOIN velocity_ceiling vc ON bm.player_full_name = vc.player_full_name
    ORDER BY bm.player_full_name
    """

    params = [
        bigquery.ScalarQueryParameter("dataset_id",      "STRING", cfg.dataset_id),
        bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
    ]

    rows = _run_sql_to_table(client, sql, dest_fq, cfg.write_disposition, params, cfg.bq_location)
    console.print(f"[green]✅ {cfg.coaching_summary_table}[/green] rows={rows:,}")
    return rows


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_velocity_intelligence(cfg: VelocityIntelConfig) -> Dict[str, int]:
    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    console.print(
        f"[bold]Velocity Intel Build[/bold] "
        f"dataset={cfg.dataset_id} version={cfg.feature_version}"
    )
    console.print(f"Run: {run_id}")

    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))

    trend_rows   = _build_session_trend(cfg, client)
    summary_rows = _build_coaching_summary(cfg, client)

    console.print(f"[bold green]DONE[/bold green] run_id={run_id}")

    return {
        "trend_rows":   trend_rows,
        "summary_rows": summary_rows,
        "run_id":       run_id,
    }


def load_cfg_from_env(dataset_id: str, feature_version: str) -> VelocityIntelConfig:
    return VelocityIntelConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        velocity_scores_table=(os.getenv("BQ_VELOCITY_SCORES_TABLE") or "scores_model_velocity").strip(),
        velocity_baseline_table=(os.getenv("BQ_VELOCITY_BASELINE_TABLE") or "baseline_model_velocity").strip(),
        session_trend_table=(os.getenv("BQ_VELOCITY_TREND_TABLE") or "velocity_session_trend_v1").strip(),
        coaching_summary_table=(os.getenv("BQ_VELOCITY_SUMMARY_TABLE") or "velocity_coaching_summary_v1").strip(),
        write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
    )
