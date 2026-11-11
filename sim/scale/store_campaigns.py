from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sim.scale.store_connection import (
    _json,
    _loads,
    _utc_stamp,
)
from sim.scale.store_models import (
    CampaignTrialResultRecord,
    SensitivityCampaignRecord,
    SensitivityTrialRecord,
)
from sim.scale.store_schema import (
    initialize_store,
)


def write_sensitivity_campaign(
    path: str | Path,
    *,
    campaign: SensitivityCampaignRecord,
    trials: list[SensitivityTrialRecord],
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    campaign_inserted = 0
    trials_inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        if _insert_sensitivity_campaign(conn, campaign=campaign, now=now):
            campaign_inserted = 1
        for trial in trials:
            if _insert_sensitivity_trial(conn, trial=trial, now=now):
                trials_inserted += 1
        conn.commit()
    return {
        "store_path": str(db_path),
        "campaigns_seen": 1,
        "campaigns_inserted": campaign_inserted,
        "trials_seen": len(trials),
        "trials_inserted": trials_inserted,
    }


def load_sensitivity_trials(
    path: str | Path,
    *,
    campaign_id: str,
    limit: int = 0,
) -> list[SensitivityTrialRecord]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    limit_clause = ""
    params: list[Any] = [campaign_id]
    if limit > 0:
        limit_clause = "LIMIT ?"
        params.append(int(limit))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                trial_id, campaign_id, trial_index, parameters_json,
                target_object_ids_json, frame_json, builder_status,
                trial_product_refs_json, input_digest, status, warnings_json
            FROM sensitivity_trials
            WHERE campaign_id = ?
            ORDER BY trial_index
            {limit_clause}
            """,
            params,
        ).fetchall()
    return [
        SensitivityTrialRecord(
            trial_id=str(row[0]),
            campaign_id=str(row[1]),
            trial_index=int(row[2]),
            parameters=dict(_loads(row[3], default={})),
            target_object_ids=tuple(str(item) for item in _loads(row[4], default=[])),
            frame=dict(_loads(row[5], default={})),
            builder_status=str(row[6] or ""),
            trial_product_refs=tuple(str(item) for item in _loads(row[7], default=[])),
            input_digest=str(row[8] or ""),
            status=str(row[9] or ""),
            warnings=tuple(str(item) for item in _loads(row[10], default=[])),
        )
        for row in rows
    ]


def load_campaign_trial_results(
    path: str | Path,
    *,
    campaign_id: str,
    run_id: str | None = None,
) -> list[CampaignTrialResultRecord]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    where = "WHERE campaign_id = ?"
    params: list[Any] = [campaign_id]
    if run_id is not None:
        where += " AND run_id = ?"
        params.append(run_id)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                result_id, run_id, campaign_id, trial_id, trial_index,
                backend, status, builder_status, metrics_json,
                product_refs_json, input_digest, error, warnings_json
            FROM campaign_trial_results
            {where}
            ORDER BY trial_index, created_utc, rowid
            """,
            params,
        ).fetchall()
    return [
        CampaignTrialResultRecord(
            result_id=str(row[0]),
            run_id=str(row[1]),
            campaign_id=str(row[2]),
            trial_id=str(row[3]),
            trial_index=int(row[4]),
            backend=str(row[5] or ""),
            status=str(row[6] or ""),
            builder_status=str(row[7] or ""),
            metrics=dict(_loads(row[8], default={})),
            product_refs=tuple(str(item) for item in _loads(row[9], default=[])),
            input_digest=str(row[10] or ""),
            error=str(row[11] or ""),
            warnings=tuple(str(item) for item in _loads(row[12], default=[])),
        )
        for row in rows
    ]


def write_campaign_trial_results(
    path: str | Path,
    *,
    results: list[CampaignTrialResultRecord],
    replace_existing: bool = False,
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for result in results:
            if _insert_campaign_trial_result(
                conn,
                result=result,
                now=now,
                replace_existing=replace_existing,
            ):
                inserted += 1
        conn.commit()
    return {
        "store_path": str(db_path),
        "results_seen": len(results),
        "results_inserted": inserted,
    }
def _insert_sensitivity_campaign(
    conn: sqlite3.Connection,
    *,
    campaign: SensitivityCampaignRecord,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO sensitivity_campaigns (
                campaign_id, workflow_name, campaign_type, target,
                uncertainty_interpretation, seed, requested_trial_count,
                realized_trial_count, sweep_mode, parameters_json, frame_json,
                trial_builder_json, outputs_json, output_dir, input_digest,
                status, warnings_json, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                campaign.campaign_id,
                campaign.workflow_name,
                campaign.campaign_type,
                campaign.target,
                campaign.uncertainty_interpretation,
                campaign.seed,
                campaign.requested_trial_count,
                campaign.realized_trial_count,
                campaign.sweep_mode,
                _json(list(campaign.parameters)),
                _json(campaign.frame),
                _json(campaign.trial_builder),
                _json(campaign.outputs),
                campaign.output_dir,
                campaign.input_digest,
                campaign.status,
                _json(list(campaign.warnings)),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        existing = conn.execute(
            "SELECT input_digest FROM sensitivity_campaigns WHERE campaign_id = ?",
            (campaign.campaign_id,),
        ).fetchone()
        if existing is not None and str(existing[0]) == campaign.input_digest:
            return False
        raise


def _insert_sensitivity_trial(
    conn: sqlite3.Connection,
    *,
    trial: SensitivityTrialRecord,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO sensitivity_trials (
                trial_id, campaign_id, trial_index, parameters_json,
                target_object_ids_json, frame_json, builder_status,
                trial_product_refs_json, input_digest, status, warnings_json,
                created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial.trial_id,
                trial.campaign_id,
                trial.trial_index,
                _json(trial.parameters),
                _json(list(trial.target_object_ids)),
                _json(trial.frame),
                trial.builder_status,
                _json(list(trial.trial_product_refs)),
                trial.input_digest,
                trial.status,
                _json(list(trial.warnings)),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        existing = conn.execute(
            "SELECT input_digest FROM sensitivity_trials WHERE trial_id = ?",
            (trial.trial_id,),
        ).fetchone()
        if existing is not None and str(existing[0]) == trial.input_digest:
            return False
        raise


def _insert_campaign_trial_result(
    conn: sqlite3.Connection,
    *,
    result: CampaignTrialResultRecord,
    now: str,
    replace_existing: bool = False,
) -> bool:
    try:
        statement = """
            INSERT INTO campaign_trial_results (
                result_id, run_id, campaign_id, trial_id, trial_index,
                backend, status, builder_status, metrics_json,
                product_refs_json, input_digest, error, warnings_json,
                created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        if replace_existing:
            statement += """
            ON CONFLICT(result_id) DO UPDATE SET
                run_id=excluded.run_id,
                campaign_id=excluded.campaign_id,
                trial_id=excluded.trial_id,
                trial_index=excluded.trial_index,
                backend=excluded.backend,
                status=excluded.status,
                builder_status=excluded.builder_status,
                metrics_json=excluded.metrics_json,
                product_refs_json=excluded.product_refs_json,
                input_digest=excluded.input_digest,
                error=excluded.error,
                warnings_json=excluded.warnings_json,
                created_utc=excluded.created_utc
            """
        conn.execute(
            statement,
            (
                result.result_id,
                result.run_id,
                result.campaign_id,
                result.trial_id,
                result.trial_index,
                result.backend,
                result.status,
                result.builder_status,
                _json(result.metrics),
                _json(list(result.product_refs)),
                result.input_digest,
                result.error,
                _json(list(result.warnings)),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        existing = conn.execute(
            "SELECT input_digest FROM campaign_trial_results WHERE result_id = ?",
            (result.result_id,),
        ).fetchone()
        if existing is not None and str(existing[0]) == result.input_digest:
            return False
        raise
