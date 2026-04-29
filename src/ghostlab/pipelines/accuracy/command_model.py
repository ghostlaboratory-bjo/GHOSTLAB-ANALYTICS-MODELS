from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from rich.console import Console
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ghostlab.io.bq import BQConfig, bq_client

console = Console()

NUMERIC_FEATURES = [
    "pitch_velocity",
    "RelHeight",
    "RelSide",
    "Extension",
    "VertRelAngle",
    "HorzRelAngle",
    "SpinRate",
    "SpinAxis",
    "VertBreak",
    "InducedVertBreak",
    "HorzBreak",
    "VertApprAngle",
    "HorzApprAngle",
    "player_weight_lb",
    "accel_impulse_lb_s",
    "accel_impulse_score_sec",
    "clawback_sec",
    "decel_impulse_lb_s",
    "decel_impulse_score_sec",
    "decel_width_sec",
    "impulse_ratio_pct",
    "stride_in",
    "stride_angle_deg",
    "stride_ratio_pct",
    "xy_back_lb",
    "xy_front_lb",
    "y_back_lb",
    "y_back_score_lb_lb",
    "y_front_lb",
    "y_front_score_lb_lb",
    "y_transfer_sec",
    "yz_back_score_lb_lb",
    "yz_front_score_lb_lb",
    "yz_transfer_back_sec",
    "yz_transfer_front_sec",
    "z_back_lb",
    "z_back_score_lb_lb",
    "z_front_lb",
    "z_front_score_lb_lb",
    "z_transfer_sec",
    "rel_height_delta",
    "rel_side_delta",
    "extension_delta",
    "relspeed_delta",
    "spinrate_delta",
    "vert_rel_angle_delta",
    "horz_rel_angle_delta",
]

CATEGORICAL_FEATURES = [
    "PitcherThrows",
    "TaggedPitchType",
]


@dataclass(frozen=True)
class CommandModelConfig:
    dataset_id: str
    feature_version: str

    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_analysis_dataset: str = "analysis_model"
    pitch_core_table: str = "pitch_core_v1"

    write_bq: bool = True
    bq_out_dataset: str = "analysis_model"
    bq_out_table: str = "model_command_scores_v1"
    bq_write_disposition: str = "WRITE_TRUNCATE"

    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 12
    min_samples_leaf: int = 5


def _pitch_core_fq(cfg: CommandModelConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{cfg.pitch_core_table}"


def _out_fq(cfg: CommandModelConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_out_dataset}.{cfg.bq_out_table}"


def _fetch_training_data(cfg: CommandModelConfig) -> pd.DataFrame:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))

    select_cols = [
        "dataset_id",
        "feature_version",
        "team_id",
        "PitchID",
        "nf_file_name",
        "event_ts",
        "session_date",
        "player_full_name",
        "PitcherId",
        "PitcherThrows",
        "TaggedPitchType",
        "plate_x_ft",
        "plate_z_ft",
        "arm_side_miss_ft",
        "outside_x_in",
        "outside_z_in",
        "miss_distance_in",
        "is_strike_target",
        "horizontal_miss_type",
        "vertical_miss_type",
        *NUMERIC_FEATURES,
    ]

    select_sql = ",\n      ".join(select_cols)

    sql = f"""
    SELECT
      {select_sql}
    FROM `{_pitch_core_fq(cfg)}`
    WHERE dataset_id = @dataset_id
      AND feature_version = @feature_version
      AND is_real_pitch = TRUE
      AND has_trackman = TRUE
      AND has_newtforce = TRUE
      AND miss_distance_in IS NOT NULL
      AND TaggedPitchType IS NOT NULL
    """

    job = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
        ]
    )

    df = client.query(sql, job_config=job).to_dataframe()
    return df


def _prep_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    work = df.copy()

    for col in NUMERIC_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    y = pd.to_numeric(work["miss_distance_in"], errors="coerce")

    x_num = work[NUMERIC_FEATURES].copy()
    x_num = x_num.replace([np.inf, -np.inf], np.nan)
    x_num = x_num.fillna(x_num.median(numeric_only=True))

    x_cat = pd.get_dummies(
        work[CATEGORICAL_FEATURES].fillna("UNKNOWN").astype(str),
        columns=CATEGORICAL_FEATURES,
        dummy_na=False,
    )

    X = pd.concat([x_num, x_cat], axis=1)

    valid = y.notna() & np.isfinite(y)
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    feature_cols = list(X.columns)
    return X, y, feature_cols


def _command_score_from_error(predicted_miss_in: pd.Series) -> pd.Series:
    """
    Coach-facing 0-100 score.
    V1 formula:
      100 = perfect command
      roughly -4 points per inch of predicted miss
    """
    score = 100.0 - (predicted_miss_in.astype(float) * 4.0)
    return score.clip(lower=0.0, upper=100.0)


def _write_df_to_bq(cfg: CommandModelConfig, df: pd.DataFrame) -> None:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    job = client.load_table_from_dataframe(
        df,
        _out_fq(cfg),
        job_config=bigquery.LoadJobConfig(write_disposition=cfg.bq_write_disposition),
    )
    job.result()


def train_command_model(cfg: CommandModelConfig) -> Dict[str, float | int | str]:
    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    model_version = "command_rf_v1"

    console.print(f"[bold]Command Model v1[/bold] dataset={cfg.dataset_id} version={cfg.feature_version}")
    console.print(f"Run: {run_id}")

    df = _fetch_training_data(cfg)
    if df.empty:
        raise RuntimeError("No eligible training rows found.")

    console.print(f"Rows fetched: {len(df):,}")

    X, y, feature_cols = _prep_features(df)
    if len(X) < 100:
        raise RuntimeError(f"Not enough rows to train command model. Rows={len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
    )

    model = RandomForestRegressor(
        n_estimators=int(cfg.n_estimators),
        max_depth=int(cfg.max_depth),
        min_samples_leaf=int(cfg.min_samples_leaf),
        random_state=int(cfg.random_state),
        n_jobs=-1,
    )

    console.print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))

    console.print(f"[green]Model metrics[/green] MAE={mae:.3f} in | R2={r2:.3f}")

    pred_all = pd.Series(model.predict(X), index=X.index, name="predicted_miss_distance_in")
    command_score = _command_score_from_error(pred_all)

    out = df.loc[X.index].copy()

    out_df = pd.DataFrame(
        {
            "dataset_id": out["dataset_id"].astype(str),
            "feature_version": out["feature_version"].astype(str),
            "model_version": model_version,
            "run_id": run_id,
            "team_id": out["team_id"].astype(str),
            "PitchID": out["PitchID"].astype(str),
            "nf_file_name": out["nf_file_name"].astype(str),
            "event_ts": out["event_ts"],
            "session_date": out["session_date"],
            "player_full_name": out["player_full_name"].astype(str),
            "PitcherId": pd.to_numeric(out["PitcherId"], errors="coerce").astype("Int64"),
            "PitcherThrows": out["PitcherThrows"].astype(str),
            "TaggedPitchType": out["TaggedPitchType"].astype(str),
            "actual_miss_distance_in": pd.to_numeric(out["miss_distance_in"], errors="coerce"),
            "predicted_miss_distance_in": pred_all.astype(float),
            "command_score": command_score.astype(float),
            "is_strike_target": pd.to_numeric(out["is_strike_target"], errors="coerce").astype("Int64"),
            "plate_x_ft": pd.to_numeric(out["plate_x_ft"], errors="coerce"),
            "plate_z_ft": pd.to_numeric(out["plate_z_ft"], errors="coerce"),
            "arm_side_miss_ft": pd.to_numeric(out["arm_side_miss_ft"], errors="coerce"),
            "outside_x_in": pd.to_numeric(out["outside_x_in"], errors="coerce"),
            "outside_z_in": pd.to_numeric(out["outside_z_in"], errors="coerce"),
            "horizontal_miss_type": out["horizontal_miss_type"].astype(str),
            "vertical_miss_type": out["vertical_miss_type"].astype(str),
            "model_mae_in": mae,
            "model_r2": r2,
            "created_at": created_at,
        }
    )

    if cfg.write_bq:
        _write_df_to_bq(cfg, out_df)
        console.print(f"[green]✅ Wrote[/green] rows={len(out_df):,} table={_out_fq(cfg)}")

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    console.print("[bold]Top features[/bold]")
    console.print(importances.head(15).to_string(index=False))

    return {
        "rows_fetched": int(len(df)),
        "rows_modeled": int(len(out_df)),
        "mae_in": mae,
        "r2": r2,
        "run_id": run_id,
        "out_table": _out_fq(cfg),
    }


def load_cfg_from_env(dataset_id: str, feature_version: str) -> CommandModelConfig:
    return CommandModelConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        pitch_core_table=(os.getenv("BQ_PITCH_CORE_TABLE") or "pitch_core_v1").strip(),
        bq_out_dataset=(os.getenv("BQ_OUT_DATASET") or "analysis_model").strip(),
        bq_out_table=(os.getenv("BQ_COMMAND_MODEL_TABLE") or "model_command_scores_v1").strip(),
        bq_write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
    )