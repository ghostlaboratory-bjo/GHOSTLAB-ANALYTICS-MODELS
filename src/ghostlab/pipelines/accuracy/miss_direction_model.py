# src/ghostlab/pipelines/accuracy/miss_direction_model.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
class MissDirectionModelConfig:
    dataset_id: str
    feature_version: str

    gcp_project: str = ""
    bq_location: str = "us-east5"
    bq_analysis_dataset: str = "analysis_model"
    pitch_core_table: str = "pitch_core_v1"

    write_bq: bool = True
    bq_out_dataset: str = "analysis_model"
    bq_out_table: str = "model_miss_direction_v1"
    bq_write_disposition: str = "WRITE_TRUNCATE"

    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 14
    min_samples_leaf: int = 5


def _pitch_core_fq(cfg: MissDirectionModelConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_analysis_dataset}.{cfg.pitch_core_table}"


def _out_fq(cfg: MissDirectionModelConfig) -> str:
    return f"{cfg.gcp_project}.{cfg.bq_out_dataset}.{cfg.bq_out_table}"


def _fetch_training_data(cfg: MissDirectionModelConfig) -> pd.DataFrame:
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

    sql = f"""
    SELECT
      {", ".join(select_cols)}
    FROM `{_pitch_core_fq(cfg)}`
    WHERE dataset_id = @dataset_id
      AND feature_version = @feature_version
      AND is_real_pitch = TRUE
      AND has_trackman = TRUE
      AND has_newtforce = TRUE
      AND miss_distance_in IS NOT NULL
      AND horizontal_miss_type IS NOT NULL
      AND vertical_miss_type IS NOT NULL
      AND TaggedPitchType IS NOT NULL
    """

    job = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", cfg.dataset_id),
            bigquery.ScalarQueryParameter("feature_version", "STRING", cfg.feature_version),
        ]
    )

    return client.query(sql, job_config=job).to_dataframe()


def _prep_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()

    for col in NUMERIC_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    x_num = work[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    x_num = x_num.fillna(x_num.median(numeric_only=True))

    x_cat = pd.get_dummies(
        work[CATEGORICAL_FEATURES].fillna("UNKNOWN").astype(str),
        columns=CATEGORICAL_FEATURES,
        dummy_na=False,
    )

    X = pd.concat([x_num, x_cat], axis=1)
    return X, list(X.columns)


def _train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: MissDirectionModelConfig,
    label_name: str,
) -> Tuple[RandomForestClassifier, float]:
    valid = y.notna() & (y.astype(str).str.len() > 0)
    Xv = X.loc[valid].copy()
    yv = y.loc[valid].astype(str).copy()

    if yv.nunique() < 2:
        raise RuntimeError(f"Not enough classes to train {label_name}. Classes={yv.unique().tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        Xv,
        yv,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        stratify=yv,
    )

    model = RandomForestClassifier(
        n_estimators=int(cfg.n_estimators),
        max_depth=int(cfg.max_depth),
        min_samples_leaf=int(cfg.min_samples_leaf),
        random_state=int(cfg.random_state),
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    console.print(f"Training {label_name} RandomForestClassifier...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    console.print(f"[green]{label_name} accuracy[/green] {acc:.3f}")
    console.print(classification_report(y_test, pred, zero_division=0))

    return model, acc


def _predict_with_confidence(model: RandomForestClassifier, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(X)
    classes = model.classes_
    max_idx = np.argmax(probs, axis=1)
    pred = classes[max_idx]
    conf = probs[np.arange(len(X)), max_idx]
    return pred, conf


def _write_df_to_bq(cfg: MissDirectionModelConfig, df: pd.DataFrame) -> None:
    client = bq_client(BQConfig(project=cfg.gcp_project, location=cfg.bq_location))
    job = client.load_table_from_dataframe(
        df,
        _out_fq(cfg),
        job_config=bigquery.LoadJobConfig(write_disposition=cfg.bq_write_disposition),
    )
    job.result()


def train_miss_direction_model(cfg: MissDirectionModelConfig) -> Dict[str, float | int | str]:
    created_at = datetime.now(timezone.utc)
    run_id = f"{cfg.dataset_id}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    model_version = "miss_direction_rf_v1"

    console.print(f"[bold]Miss Direction Model v1[/bold] dataset={cfg.dataset_id} version={cfg.feature_version}")
    console.print(f"Run: {run_id}")

    df = _fetch_training_data(cfg)
    if df.empty:
        raise RuntimeError("No eligible rows found for miss direction model.")

    console.print(f"Rows fetched: {len(df):,}")

    X, feature_cols = _prep_features(df)

    h_model, h_acc = _train_classifier(
        X,
        df["horizontal_miss_type"],
        cfg,
        "horizontal_miss_type",
    )

    v_model, v_acc = _train_classifier(
        X,
        df["vertical_miss_type"],
        cfg,
        "vertical_miss_type",
    )

    h_pred, h_conf = _predict_with_confidence(h_model, X)
    v_pred, v_conf = _predict_with_confidence(v_model, X)

    out_df = pd.DataFrame(
        {
            "dataset_id": df["dataset_id"].astype(str),
            "feature_version": df["feature_version"].astype(str),
            "model_version": model_version,
            "run_id": run_id,
            "team_id": df["team_id"].astype(str),
            "PitchID": df["PitchID"].astype(str),
            "nf_file_name": df["nf_file_name"].astype(str),
            "event_ts": df["event_ts"],
            "session_date": df["session_date"],
            "player_full_name": df["player_full_name"].astype(str),
            "PitcherId": pd.to_numeric(df["PitcherId"], errors="coerce").astype("Int64"),
            "PitcherThrows": df["PitcherThrows"].astype(str),
            "TaggedPitchType": df["TaggedPitchType"].astype(str),
            "actual_horizontal_miss_type": df["horizontal_miss_type"].astype(str),
            "predicted_horizontal_miss_type": pd.Series(h_pred).astype(str),
            "horizontal_confidence": pd.Series(h_conf).astype(float),
            "actual_vertical_miss_type": df["vertical_miss_type"].astype(str),
            "predicted_vertical_miss_type": pd.Series(v_pred).astype(str),
            "vertical_confidence": pd.Series(v_conf).astype(float),
            "miss_distance_in": pd.to_numeric(df["miss_distance_in"], errors="coerce"),
            "is_strike_target": pd.to_numeric(df["is_strike_target"], errors="coerce").astype("Int64"),
            "plate_x_ft": pd.to_numeric(df["plate_x_ft"], errors="coerce"),
            "plate_z_ft": pd.to_numeric(df["plate_z_ft"], errors="coerce"),
            "arm_side_miss_ft": pd.to_numeric(df["arm_side_miss_ft"], errors="coerce"),
            "outside_x_in": pd.to_numeric(df["outside_x_in"], errors="coerce"),
            "outside_z_in": pd.to_numeric(df["outside_z_in"], errors="coerce"),
            "horizontal_accuracy": h_acc,
            "vertical_accuracy": v_acc,
            "created_at": created_at,
        }
    )

    if cfg.write_bq:
        _write_df_to_bq(cfg, out_df)
        console.print(f"[green]✅ Wrote[/green] rows={len(out_df):,} table={_out_fq(cfg)}")

    h_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": h_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    v_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": v_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    console.print("[bold]Top horizontal miss features[/bold]")
    console.print(h_imp.head(12).to_string(index=False))

    console.print("[bold]Top vertical miss features[/bold]")
    console.print(v_imp.head(12).to_string(index=False))

    return {
        "rows_fetched": int(len(df)),
        "rows_modeled": int(len(out_df)),
        "horizontal_accuracy": h_acc,
        "vertical_accuracy": v_acc,
        "run_id": run_id,
        "out_table": _out_fq(cfg),
    }


def load_cfg_from_env(dataset_id: str, feature_version: str) -> MissDirectionModelConfig:
    return MissDirectionModelConfig(
        dataset_id=dataset_id,
        feature_version=feature_version,
        gcp_project=(os.getenv("GCP_PROJECT") or "").strip(),
        bq_location=(os.getenv("BQ_LOCATION") or "us-east5").strip(),
        bq_analysis_dataset=(os.getenv("BQ_ANALYSIS_DATASET") or "analysis_model").strip(),
        pitch_core_table=(os.getenv("BQ_PITCH_CORE_TABLE") or "pitch_core_v1").strip(),
        bq_out_dataset=(os.getenv("BQ_OUT_DATASET") or "analysis_model").strip(),
        bq_out_table=(os.getenv("BQ_MISS_DIRECTION_MODEL_TABLE") or "model_miss_direction_v1").strip(),
        bq_write_disposition=(os.getenv("BQ_WRITE_DISPOSITION") or "WRITE_TRUNCATE").strip(),
    )