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
from ghostlab.pipelines.accuracy.command_model import (
    train_command_model,
    load_cfg_from_env as load_command_cfg_from_env,
)

from ghostlab.pipelines.accuracy.miss_direction_model import (
    train_miss_direction_model,
    load_cfg_from_env as load_miss_direction_cfg_from_env,
)

from ghostlab.pipelines.intelligence.pitch_intelligence import (
    build_pitch_intelligence,
    load_cfg_from_env as load_intel_cfg_from_env,
)
from ghostlab.pipelines.baselines.nf_velocity_pattern import (
    build_velocity_baselines,
    load_cfg_from_env as load_velocity_baseline_cfg_from_env,
)
from ghostlab.pipelines.scores.nf_velocity_scoring import (
    score_velocity_pitches,
    load_cfg_from_env as load_velocity_score_cfg_from_env,
)
from ghostlab.pipelines.intelligence.velocity_intelligence import (
    build_velocity_intelligence,
    load_cfg_from_env as load_velocity_intel_cfg_from_env,
)
from ghostlab.pipelines.intelligence.velocity_nf_analysis import (
    build_velocity_nf_analysis,
    load_cfg_from_env as load_velocity_nf_analysis_cfg_from_env,
)

console = Console()

DEFAULT_BQ_DISPOSITION = "WRITE_TRUNCATE"


def cmd_doctor(_args: argparse.Namespace) -> int:
    console.print("[green]OK:[/green] ghostlab CLI is wired up.")
    console.print(f"Repo root: {settings.repo_root}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    cfg = load_baseline_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = asdict(cfg)

    if args.mode:
        overrides["mode"] = args.mode
    if args.target_timesteps is not None:
        overrides["target_timesteps"] = int(args.target_timesteps)
    if args.max_pitches_per_group is not None:
        overrides["max_pitches_per_group"] = int(args.max_pitches_per_group)
    if args.pitch_types:
        overrides["pitch_types"] = tuple(p.strip() for p in args.pitch_types.split(",") if p.strip())

    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION
        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset
        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition
        if args.bq_baseline_table_base:
            overrides["bq_baseline_table_base"] = args.bq_baseline_table_base

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

    if args.mode:
        overrides["mode"] = args.mode
    if args.target_timesteps is not None:
        overrides["target_timesteps"] = int(args.target_timesteps)
    if args.pitch_types:
        overrides["pitch_types"] = tuple(p.strip() for p in args.pitch_types.split(",") if p.strip())
    if args.baseline_run_id:
        overrides["baseline_run_id"] = args.baseline_run_id

    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION
        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset
        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition
        if args.bq_scores_table_base:
            overrides["bq_scores_table_base"] = args.bq_scores_table_base

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


def cmd_command_model(args: argparse.Namespace) -> int:
    cfg = load_command_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = asdict(cfg)

    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION
        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset
        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition

    if args.gcp_project:
        overrides["gcp_project"] = args.gcp_project
    if args.bq_location:
        overrides["bq_location"] = args.bq_location
    if args.bq_analysis_dataset:
        overrides["bq_analysis_dataset"] = args.bq_analysis_dataset
    if args.pitch_core_table:
        overrides["pitch_core_table"] = args.pitch_core_table

    cfg = cfg.__class__(**overrides)
    stats = train_command_model(cfg)
    console.print(f"[bold green]DONE[/bold green] command-model stats={stats}")
    return 0


def cmd_miss_direction_model(args: argparse.Namespace) -> int:
    cfg = load_miss_direction_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = asdict(cfg)

    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION

        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset

        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition

    if args.gcp_project:
        overrides["gcp_project"] = args.gcp_project
    if args.bq_location:
        overrides["bq_location"] = args.bq_location
    if args.bq_analysis_dataset:
        overrides["bq_analysis_dataset"] = args.bq_analysis_dataset
    if args.pitch_core_table:
        overrides["pitch_core_table"] = args.pitch_core_table

    cfg = cfg.__class__(**overrides)

    stats = train_miss_direction_model(cfg)
    console.print(f"[bold green]DONE[/bold green] miss-direction stats={stats}")
    return 0

def cmd_intel_build(args: argparse.Namespace) -> int:
    cfg = load_intel_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    stats = build_pitch_intelligence(cfg)
    console.print(f"[bold green]DONE[/bold green] intel-build stats={stats}")
    return 0


def cmd_velocity_baseline(args: argparse.Namespace) -> int:
    cfg = load_velocity_baseline_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = {
        "dataset_id":      cfg.dataset_id,
        "feature_version": cfg.feature_version,
        "target_timesteps":       cfg.target_timesteps,
        "top_velocity_pct":       cfg.top_velocity_pct,
        "min_pitches_for_pattern": cfg.min_pitches_for_pattern,
        "gcp_project":            cfg.gcp_project,
        "bq_location":            cfg.bq_location,
        "bq_src_dataset":         cfg.bq_src_dataset,
        "bq_analysis_dataset":    cfg.bq_analysis_dataset,
        "pitch_core_table":       cfg.pitch_core_table,
        "nf_time_series_table":   cfg.nf_time_series_table,
        "write_bq":               cfg.write_bq,
        "bq_out_dataset":         cfg.bq_out_dataset,
        "bq_write_disposition":   cfg.bq_write_disposition,
        "bq_baseline_table":      cfg.bq_baseline_table,
        "local_root":             cfg.local_root,
        "write_batch_rows":       cfg.write_batch_rows,
        "file_batch_size":        cfg.file_batch_size,
    }

    if args.top_velocity_pct is not None:
        overrides["top_velocity_pct"] = float(args.top_velocity_pct)
    if args.min_pitches is not None:
        overrides["min_pitches_for_pattern"] = int(args.min_pitches)

    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION
        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset
        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition

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
    out_path = build_velocity_baselines(cfg)
    console.print(f"[bold green]DONE[/bold green] {out_path}")
    return 0


def cmd_velocity_intel(args: argparse.Namespace) -> int:
    cfg = load_velocity_intel_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = {
        "dataset_id":              cfg.dataset_id,
        "feature_version":         cfg.feature_version,
        "gcp_project":             cfg.gcp_project,
        "bq_location":             cfg.bq_location,
        "bq_analysis_dataset":     cfg.bq_analysis_dataset,
        "velocity_scores_table":   cfg.velocity_scores_table,
        "velocity_baseline_table": cfg.velocity_baseline_table,
        "session_trend_table":     cfg.session_trend_table,
        "coaching_summary_table":  cfg.coaching_summary_table,
        "write_disposition":       cfg.write_disposition,
    }

    if args.gcp_project:
        overrides["gcp_project"] = args.gcp_project
    if args.bq_location:
        overrides["bq_location"] = args.bq_location
    if args.bq_analysis_dataset:
        overrides["bq_analysis_dataset"] = args.bq_analysis_dataset

    cfg = cfg.__class__(**overrides)
    stats = build_velocity_intelligence(cfg)
    console.print(f"[bold green]DONE[/bold green] velocity-intel stats={stats}")
    return 0


def cmd_velocity_score(args: argparse.Namespace) -> int:
    cfg = load_velocity_score_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    overrides = {
        "dataset_id":          cfg.dataset_id,
        "feature_version":     cfg.feature_version,
        "target_timesteps":    cfg.target_timesteps,
        "baseline_run_id":     cfg.baseline_run_id,
        "gcp_project":         cfg.gcp_project,
        "bq_location":         cfg.bq_location,
        "bq_src_dataset":      cfg.bq_src_dataset,
        "bq_analysis_dataset": cfg.bq_analysis_dataset,
        "pitch_core_table":    cfg.pitch_core_table,
        "nf_time_series_table": cfg.nf_time_series_table,
        "bq_baseline_table":   cfg.bq_baseline_table,
        "write_bq":            cfg.write_bq,
        "bq_out_dataset":      cfg.bq_out_dataset,
        "bq_write_disposition": cfg.bq_write_disposition,
        "bq_scores_table":     cfg.bq_scores_table,
        "file_batch_size":     cfg.file_batch_size,
        "write_batch_rows":    cfg.write_batch_rows,
    }

    if args.baseline_run_id:
        overrides["baseline_run_id"] = args.baseline_run_id

    overrides["write_bq"] = bool(args.write_bq)

    if args.write_bq:
        if not args.bq_write_disposition:
            overrides["bq_write_disposition"] = DEFAULT_BQ_DISPOSITION
        if args.bq_out_dataset:
            overrides["bq_out_dataset"] = args.bq_out_dataset
        if args.bq_write_disposition:
            overrides["bq_write_disposition"] = args.bq_write_disposition

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
    stats = score_velocity_pitches(cfg)
    console.print(f"[bold green]DONE[/bold green] velocity-score stats={stats}")
    return 0


def cmd_velocity_nf_analysis(args: argparse.Namespace) -> int:
    cfg = load_velocity_nf_analysis_cfg_from_env(dataset_id=args.dataset, feature_version=args.version)
    from dataclasses import asdict
    overrides = asdict(cfg)

    if args.gcp_project:
        overrides["gcp_project"] = args.gcp_project
    if args.bq_location:
        overrides["bq_location"] = args.bq_location
    if args.bq_analysis_dataset:
        overrides["bq_analysis_dataset"] = args.bq_analysis_dataset
    if args.bq_src_dataset:
        overrides["bq_src_dataset"] = args.bq_src_dataset

    cfg = cfg.__class__(**overrides)
    n_rows = build_velocity_nf_analysis(cfg)
    console.print(f"[bold green]DONE[/bold green] velocity-nf-analysis rows={n_rows}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ghostlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="Sanity check installation")
    p_doctor.set_defaults(func=cmd_doctor)

    # ---------------- baseline ----------------
    p_base = sub.add_parser("baseline", help="Build NewtForce baseline waveforms")
    p_base.add_argument("--dataset", required=True)
    p_base.add_argument("--version", default="v1")
    p_base.add_argument("--mode", choices=["velocity", "accuracy"])
    p_base.add_argument("--target-timesteps", type=int)
    p_base.add_argument("--max-pitches-per-group", type=int)
    p_base.add_argument("--pitch-types", default=None)

    p_base.add_argument("--write-bq", action="store_true")
    p_base.add_argument("--bq-out-dataset", default=None)
    p_base.add_argument(
        "--bq-write-disposition",
        default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
    )
    p_base.add_argument("--bq-baseline-table-base", default=None)

    p_base.add_argument("--gcp-project", default=None)
    p_base.add_argument("--bq-location", default=None)
    p_base.add_argument("--bq-src-dataset", default=None)
    p_base.add_argument("--bq-analysis-dataset", default=None)
    p_base.add_argument("--pitch-core-table", default=None)
    p_base.add_argument("--nf-time-series-table", default=None)
    p_base.set_defaults(func=cmd_baseline)

    # ---------------- score-nf ----------------
    p_score = sub.add_parser("score-nf", help="Score pitches vs NF baselines")
    p_score.add_argument("--dataset", required=True)
    p_score.add_argument("--version", default="v1")
    p_score.add_argument("--mode", choices=["velocity", "accuracy"], required=True)
    p_score.add_argument("--target-timesteps", type=int)
    p_score.add_argument("--pitch-types", default=None)
    p_score.add_argument("--baseline-run-id", default=None)

    p_score.add_argument("--write-bq", action="store_true")
    p_score.add_argument("--bq-out-dataset", default=None)
    p_score.add_argument(
        "--bq-write-disposition",
        default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
    )
    p_score.add_argument("--bq-scores-table-base", default=None)

    p_score.add_argument("--gcp-project", default=None)
    p_score.add_argument("--bq-location", default=None)
    p_score.add_argument("--bq-src-dataset", default=None)
    p_score.add_argument("--bq-analysis-dataset", default=None)
    p_score.add_argument("--pitch-core-table", default=None)
    p_score.add_argument("--nf-time-series-table", default=None)
    p_score.set_defaults(func=cmd_score_nf)

    # ---------------- command-model ----------------
    p_cmd = sub.add_parser("command-model", help="Train command / accuracy model")
    p_cmd.add_argument("--dataset", required=True)
    p_cmd.add_argument("--version", default="v1")

    p_cmd.add_argument("--write-bq", action="store_true")
    p_cmd.add_argument("--bq-out-dataset", default=None)
    p_cmd.add_argument(
        "--bq-write-disposition",
        default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
    )

    p_cmd.add_argument("--gcp-project", default=None)
    p_cmd.add_argument("--bq-location", default=None)
    p_cmd.add_argument("--bq-analysis-dataset", default=None)
    p_cmd.add_argument("--pitch-core-table", default=None)
    p_cmd.set_defaults(func=cmd_command_model)

    # ---------------- miss-direction ----------------
    p_md = sub.add_parser("miss-direction", help="Train miss direction model")
    p_md.add_argument("--dataset", required=True)
    p_md.add_argument("--version", default="v1")

    p_md.add_argument("--write-bq", action="store_true")
    p_md.add_argument("--bq-out-dataset", default=None)
    p_md.add_argument(
        "--bq-write-disposition",
        default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
    )

    p_md.add_argument("--gcp-project", default=None)
    p_md.add_argument("--bq-location", default=None)
    p_md.add_argument("--bq-analysis-dataset", default=None)
    p_md.add_argument("--pitch-core-table", default=None)

    p_md.set_defaults(func=cmd_miss_direction_model)

    # ---------------- intel-build ----------------
    p_intel = sub.add_parser("intel-build", help="Build Coach Intelligence BQ tables (4 tables in sequence)")
    p_intel.add_argument("--dataset", required=True)
    p_intel.add_argument("--version", default="v1")
    # BQ connection args accepted (read from env vars, but parsed to avoid argparse errors)
    p_intel.add_argument("--gcp-project",        default=None)
    p_intel.add_argument("--bq-location",        default=None)
    p_intel.add_argument("--bq-analysis-dataset", default=None)
    p_intel.set_defaults(func=cmd_intel_build)

    # ---------------- velocity-intel ----------------
    p_vintel = sub.add_parser("velocity-intel", help="Build velocity coaching intelligence tables")
    p_vintel.add_argument("--dataset",              required=True)
    p_vintel.add_argument("--version",              default="v1")
    p_vintel.add_argument("--gcp-project",          default=None)
    p_vintel.add_argument("--bq-location",          default=None)
    p_vintel.add_argument("--bq-analysis-dataset",  default=None)
    p_vintel.set_defaults(func=cmd_velocity_intel)

    # ---------------- velocity-baseline ----------------
    p_vbase = sub.add_parser("velocity-baseline", help="Build velocity max-effort NF baselines (FB+SI top 10%%)")
    p_vbase.add_argument("--dataset",           required=True)
    p_vbase.add_argument("--version",           default="v1")
    p_vbase.add_argument("--top-velocity-pct",  type=float, default=None,
                         help="Fraction of highest-velocity pitches to use (default 0.10)")
    p_vbase.add_argument("--min-pitches",       type=int, default=None,
                         help="Min pitches required to build a baseline (default 5)")

    p_vbase.add_argument("--write-bq",          action="store_true")
    p_vbase.add_argument("--bq-out-dataset",    default=None)
    p_vbase.add_argument(
        "--bq-write-disposition", default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
    )

    p_vbase.add_argument("--gcp-project",             default=None)
    p_vbase.add_argument("--bq-location",             default=None)
    p_vbase.add_argument("--bq-src-dataset",          default=None)
    p_vbase.add_argument("--bq-analysis-dataset",     default=None)
    p_vbase.add_argument("--pitch-core-table",        default=None)
    p_vbase.add_argument("--nf-time-series-table",    default=None)
    p_vbase.set_defaults(func=cmd_velocity_baseline)

    # ---------------- velocity-nf-analysis ----------------
    p_vna = sub.add_parser(
        "velocity-nf-analysis",
        help="Compute per-player NF feature correlations vs pitch velocity (FB/SI/Cutter)"
    )
    p_vna.add_argument("--dataset",              required=True)
    p_vna.add_argument("--version",              default="v1")
    p_vna.add_argument("--gcp-project",          default=None)
    p_vna.add_argument("--bq-location",          default=None)
    p_vna.add_argument("--bq-analysis-dataset",  default=None)
    p_vna.add_argument("--bq-src-dataset",       default=None)
    p_vna.set_defaults(func=cmd_velocity_nf_analysis)

    # ---------------- velocity-score ----------------
    p_vscore = sub.add_parser("velocity-score", help="Score FB+SI pitches vs velocity baselines")
    p_vscore.add_argument("--dataset",          required=True)
    p_vscore.add_argument("--version",          default="v1")
    p_vscore.add_argument("--baseline-run-id",  default=None,
                          help="Specific baseline run_id to use (default: auto-pick latest)")

    p_vscore.add_argument("--write-bq",         action="store_true")
    p_vscore.add_argument("--bq-out-dataset",   default=None)
    p_vscore.add_argument(
        "--bq-write-disposition", default=None,
        choices=["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"],
    )

    p_vscore.add_argument("--gcp-project",            default=None)
    p_vscore.add_argument("--bq-location",            default=None)
    p_vscore.add_argument("--bq-src-dataset",         default=None)
    p_vscore.add_argument("--bq-analysis-dataset",    default=None)
    p_vscore.add_argument("--pitch-core-table",       default=None)
    p_vscore.add_argument("--nf-time-series-table",   default=None)
    p_vscore.set_defaults(func=cmd_velocity_score)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())