# src/ghostlab/cli.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from ghostlab.settings import settings
from ghostlab.pipelines.baselines.nf_best_pattern import (
    build_nf_best_pattern_baselines,
    load_cfg_from_env as load_baseline_cfg_from_env,
)
from ghostlab.pipelines.scores.nf_pitch_deviation import (
    score_nf_pitches,
    load_cfg_from_env as load_score_cfg_from_env,
)

console = Console()

# Full rebuild default (matches your new “base model tables” philosophy)
DEFAULT_BQ_DISPOSITION = "WRITE_TRUNCATE"


def cmd_doctor(_args: argparse.Namespace) -> int:
    console.print("[green]OK:[/green] ghostlab CLI is wired up.")
    console.print(f"Repo root: {settings.repo_root}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    cfg = load_baseline_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = asdict(cfg)

    # ---------------- core overrides ----------------
    if args.mode:
        overrides["mode"] = args.mode
    if args.target_timesteps is not None:
        overrides["target_timesteps"] = int(args.target_timesteps)
    if args.max_pitches_per_group is not None:
        overrides["max_pitches_per_group"] = int(args.max_pitches_per_group)

    # ✅ only override if user provided pitch-types
    if args.pitch_types:
        overrides["pitch_types"] = tuple(p.strip() for p in args.pitch_types.split(",") if p.strip())

    # ---------------- BigQuery write controls ----------------
    # ✅ IMPORTANT: only write to BQ if user explicitly asked
    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        # Default to TRUNCATE unless explicitly overridden.
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION

        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset
        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition

        # Optional base-table naming (only applies if the config supports it)
        if args.bq_baseline_table_base:
            overrides["bq_baseline_table_base"] = args.bq_baseline_table_base

    # ---------------- connection/source overrides ----------------
    if args.gcp_project:
        overrides["gcp_project"] = args.gcp_project
    if args.bq_location:
        overrides["bq_location"] = args.bq_location
    if args.bq_src_dataset:
        overrides["bq_src_dataset"] = args.bq_src_dataset
    if args.bq_analysis_dataset:
        overrides["bq_analysis_dataset"] = args.bq_analysis_dataset
    if args.pitch_core_table:
        overrides["pitch_core_table"] = args.pitch_core_table
    if args.nf_time_series_table:
        overrides["nf_time_series_table"] = args.nf_time_series_table

    cfg = cfg.__class__(**overrides)

    out_path = build_nf_best_pattern_baselines(cfg)
    console.print(f"[bold green]DONE[/bold green] {out_path}")
    return 0


def cmd_score_nf(args: argparse.Namespace) -> int:
    cfg = load_score_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = asdict(cfg)

    # ---------------- core overrides ----------------
    if args.mode:
        overrides["mode"] = args.mode
    if args.target_timesteps is not None:
        overrides["target_timesteps"] = int(args.target_timesteps)

    # ✅ only override if user provided pitch-types
    if args.pitch_types:
        overrides["pitch_types"] = tuple(p.strip() for p in args.pitch_types.split(",") if p.strip())

    if args.baseline_run_id:
        overrides["baseline_run_id"] = args.baseline_run_id

    # ---------------- BigQuery write controls ----------------
    # ✅ IMPORTANT: only write to BQ if user explicitly asked
    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        # Default to TRUNCATE unless explicitly overridden.
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION

        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset
        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition

        # Optional base-table naming
        if args.bq_scores_table_base:
            overrides["bq_scores_table_base"] = args.bq_scores_table_base

    # ---------------- connection/source overrides ----------------
    if args.gcp_project:
        overrides["gcp_project"] = args.gcp_project
    if args.bq_location:
        overrides["bq_location"] = args.bq_location
    if args.bq_src_dataset:
        overrides["bq_src_dataset"] = args.bq_src_dataset
    if args.bq_analysis_dataset:
        overrides["bq_analysis_dataset"] = args.bq_analysis_dataset
    if args.pitch_core_table:
        overrides["pitch_core_table"] = args.pitch_core_table
    if args.nf_time_series_table:
        overrides["nf_time_series_table"] = args.nf_time_series_table

    cfg = cfg.__class__(**overrides)

    stats = score_nf_pitches(cfg)
    console.print(f"[bold green]DONE[/bold green] score-nf stats={stats}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ghostlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="Sanity check installation")
    p_doctor.set_defaults(func=cmd_doctor)

    # ---------------- baseline ----------------
    p_base = sub.add_parser("baseline", help="Build NewtForce baseline waveforms")
    p_base.add_argument("--dataset", required=True, help="Dataset id (e.g. alpha_bsbl)")
    p_base.add_argument("--version", default="v1", help="Feature version (default: v1)")
    p_base.add_argument("--mode", choices=["velocity", "accuracy"], help="Baseline mode")
    p_base.add_argument("--target-timesteps", type=int, help="Resample length (default from env)")
    p_base.add_argument("--max-pitches-per-group", type=int, help="Cap pitches per (player,pitch_type)")
    # ✅ default None so omission does NOT override config default (Fastball)
    p_base.add_argument("--pitch-types", default=None, help="CSV list (default: Fastball)")

    p_base.add_argument("--write-bq", action="store_true", help="Write results to BigQuery")
    p_base.add_argument("--bq-out-dataset", default=None, help="BQ dataset to write into (e.g. analysis_model)")
    p_base.add_argument(
        "--bq-write-disposition",
        default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
        help="BQ write disposition (default when --write-bq: WRITE_TRUNCATE)",
    )

    # ✅ new: allows your base model table naming
    p_base.add_argument(
        "--bq-baseline-table-base",
        default=None,
        help="Base name for baseline tables (default in code/env). Example: baseline_model -> baseline_model_velocity|accuracy",
    )

    p_base.add_argument("--gcp-project", default=None, help="Override GCP project")
    p_base.add_argument("--bq-location", default=None, help="Override BQ location (e.g. us-east5)")
    p_base.add_argument("--bq-src-dataset", default=None, help="Override source dataset (gold NF TS)")
    p_base.add_argument("--bq-analysis-dataset", default=None, help="Override analysis dataset (pitch_core)")
    p_base.add_argument("--pitch-core-table", default=None, help="Override pitch_core table name")
    p_base.add_argument("--nf-time-series-table", default=None, help="Override NF time series table name")
    p_base.set_defaults(func=cmd_baseline)

    # ---------------- score-nf ----------------
    p_score = sub.add_parser("score-nf", help="Score pitches vs NF baselines (deviation/similarity)")
    p_score.add_argument("--dataset", required=True, help="Dataset id (e.g. alpha_bsbl)")
    p_score.add_argument("--version", default="v1", help="Feature version (default: v1)")
    p_score.add_argument("--mode", choices=["velocity", "accuracy"], required=True, help="Scoring mode")
    p_score.add_argument("--target-timesteps", type=int, help="Resample length (default: 700)")
    # ✅ default None so omission does NOT override config default (Fastball)
    p_score.add_argument("--pitch-types", default=None, help="CSV list (default: Fastball)")
    p_score.add_argument("--baseline-run-id", default=None, help="Override baseline run_id (else auto-pick latest)")

    p_score.add_argument("--write-bq", action="store_true", help="Write scores to BigQuery")
    p_score.add_argument("--bq-out-dataset", default=None, help="BQ dataset to write into (e.g. analysis_model)")
    p_score.add_argument(
        "--bq-write-disposition",
        default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
        help="BQ write disposition (default when --write-bq: WRITE_TRUNCATE)",
    )

    # ✅ new: allows your base model table naming
    p_score.add_argument(
        "--bq-scores-table-base",
        default=None,
        help="Base name for scores tables (default in code/env). Example: scores_model -> scores_model_velocity|accuracy",
    )

    p_score.add_argument("--gcp-project", default=None, help="Override GCP project")
    p_score.add_argument("--bq-location", default=None, help="Override BQ location (e.g. us-east5)")
    p_score.add_argument("--bq-src-dataset", default=None, help="Override source dataset (gold NF TS)")
    p_score.add_argument("--bq-analysis-dataset", default=None, help="Override analysis dataset (pitch_core)")
    p_score.add_argument("--pitch-core-table", default=None, help="Override pitch_core table name")
    p_score.add_argument("--nf-time-series-table", default=None, help="Override NF time series table name")
    p_score.set_defaults(func=cmd_score_nf)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())