from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sim.execution import run_simulation_config_file
from sim.review import SAVED_REVIEW_QUERIES
from tools.prepare_agent_feedback import build_agent_feedback_issue

ROOT = Path(__file__).resolve().parents[2]
AGENT_EXAMPLES = [
    ROOT / "agents" / "examples" / "public_agent_python_api_minimal_propagation.yaml",
    ROOT / "agents" / "examples" / "public_agent_single_satellite.yaml",
    ROOT / "agents" / "examples" / "public_agent_ogp_sgp4_propagation.yaml",
    ROOT / "agents" / "examples" / "public_agent_ogp_sdp4_propagation.yaml",
    ROOT / "agents" / "examples" / "public_agent_rendezvous_lqr.yaml",
    ROOT / "agents" / "examples" / "public_agent_mission_recovery_plus_c_burn.yaml",
    ROOT / "agents" / "examples" / "public_agent_mission_reconstitution_trade_space.yaml",
    ROOT / "agents" / "examples" / "public_agent_ground_access.yaml",
    ROOT / "agents" / "examples" / "public_agent_attitude_hold.yaml",
]
AGENT_EXAMPLE_IDS = [path.stem for path in AGENT_EXAMPLES]
AGENT_TASK_CARDS = sorted((ROOT / "agents" / "tasks").glob("*.md"))
AGENT_TASK_CARD_IDS = [path.stem for path in AGENT_TASK_CARDS]
GOLDEN_PATHS = {
    "minimal_propagation": {
        "configs": [
            "agents/examples/public_agent_single_satellite.yaml",
            "agents/examples/public_agent_python_api_minimal_propagation.yaml",
        ],
        "queries": ["run_metadata", "objects", "passive_final_state", "artifacts"],
    },
    "ogp_sgp4_propagation": {
        "configs": ["agents/examples/public_agent_ogp_sgp4_propagation.yaml"],
        "queries": ["run_metadata", "objects", "ogp_propagation_contract", "passive_final_state", "artifacts"],
    },
    "closed_loop_rendezvous": {
        "configs": ["agents/examples/public_agent_rendezvous_lqr.yaml"],
        "queries": [
            "run_metadata",
            "rendezvous_metrics",
            "rendezvous_closest_approach",
            "relative_final_state",
            "burn_activity",
            "burn_events",
        ],
    },
    "mission_recovery_reconstitution": {
        "configs": [
            "agents/examples/public_agent_mission_recovery_plus_c_burn.yaml",
            "agents/examples/public_agent_mission_reconstitution_trade_space.yaml",
        ],
        "queries": [
            "run_metadata",
            "burn_activity",
            "mission_recovery_summary",
            "mission_recovery_elements",
            "mission_recovery_candidates",
            "mission_recovery_burns",
        ],
    },
}
ANSWER_EXAMPLE_REQUIRED_SECTIONS = [
    "Status:",
    "Commands:",
    "Review queries:",
    "Outputs inspected:",
    "Evidence:",
    "Conclusion:",
    "Limitations:",
    "Next run:",
]


def _task_card_example_path(card_path: Path) -> Path:
    text = card_path.read_text(encoding="utf-8")
    match = re.search(r"^Example config: `([^`]+)`$", text, flags=re.MULTILINE)
    assert match is not None, card_path
    return ROOT / match.group(1)


def _task_card_sql_blocks(card_path: Path) -> list[str]:
    text = card_path.read_text(encoding="utf-8")
    return [block.strip() for block in re.findall(r"```sql\n(.*?)\n```", text, flags=re.DOTALL)]


def _task_card_answer_example_path(card_path: Path) -> Path:
    text = card_path.read_text(encoding="utf-8")
    match = re.search(r"^Answer example: `([^`]+)`$", text, flags=re.MULTILINE)
    assert match is not None, card_path
    return ROOT / match.group(1)


def test_public_agent_docs_define_boundaries_and_commands() -> None:
    root_agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    agents_readme = (ROOT / "agents" / "README.md").read_text(encoding="utf-8")
    public_agents = (ROOT / "agents" / "public" / "AGENTS.md").read_text(encoding="utf-8")
    rubric = (ROOT / "agents" / "public" / "evaluation-rubric.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "oel-agents.md").read_text(encoding="utf-8")
    eval_packet = (ROOT / "docs" / "agent-evaluation-packet.md").read_text(encoding="utf-8")
    golden_paths = (ROOT / "docs" / "agent-capability-routing.md").read_text(encoding="utf-8")
    capability_routing = (ROOT / "docs" / "agent-capability-routing.md").read_text(encoding="utf-8")
    review_queries = (ROOT / "docs" / "agent-review-queries.md").read_text(encoding="utf-8")
    feedback_loop = (ROOT / "docs" / "agent-feedback-loop.md").read_text(encoding="utf-8")
    task_cards = (ROOT / "docs" / "agent-task-cards.md").read_text(encoding="utf-8")
    root_agents_normalized = " ".join(root_agents.split())

    assert "orchestrate documented workflows" in root_agents
    assert "python run_simulation.py --config <path> --validate-only" in root_agents
    assert "docs/installation.md" in root_agents
    assert "docs/agent-run-lifecycle.md" in root_agents
    assert "oel runs start" in root_agents
    assert "oel runs await" in root_agents
    assert "accepted manifest reference" in root_agents
    assert "normalized-config digest" in root_agents
    assert "arbitrary commands or release workflows" in root_agents_normalized
    assert "Scenario Generation Rules" in public_agents
    assert "scenario YAML or typed orbital-analysis problem -> validate" in public_agents
    assert "Natural User Requests" in public_agents
    assert "Agent Workflow Evaluation Rubric" in rubric
    assert "review/run.sqlite" in rubric
    assert "python run_simulation.py --config <scenario.yaml> --validate-only" in docs
    assert "Installing OEL" in docs
    assert "Natural Requests" in docs
    assert "When The User Asks Something New" in root_agents
    assert "When The User Asks Something New" in public_agents
    assert "When The User Asks Something New" in docs
    assert "Agent Example Cookbook" in docs
    assert "Evaluation Fixtures" in docs
    assert "not the conceptual boundary" in docs
    assert "OEL Agents" in docs
    assert "not" in task_cards
    assert "boundary of what OEL Agents can help users do" in task_cards
    assert "scenario YAML or typed orbital-analysis problem -> validate" in eval_packet
    assert "oel://analysis/workflows/v1" in root_agents
    assert "authoritative replay" in public_agents
    assert "Agent Review Query Recipes" in review_queries
    assert "SELECT scenario_name, duration_s, dt_s, samples" in review_queries
    assert "--saved-query rendezvous_metrics" in review_queries
    assert "--saved-query attitude_state_first_last" in review_queries
    assert "agent-feedback-loop.md" in root_agents
    assert "agent-feedback-loop.md" in public_agents
    assert "agent-feedback-loop.md" in docs
    assert "Agent Feedback issue template" in feedback_loop
    assert "Agents must never submit feedback silently" in feedback_loop
    assert "tools/prepare_agent_feedback.py" in feedback_loop
    assert "agent-task-cards.md" in root_agents
    assert "agent-task-cards.md" in public_agents
    assert "agent-task-cards.md" in docs
    assert "agent-capability-routing.md" in agents_readme
    assert "agent-capability-routing.md" in public_agents
    assert "agent-capability-routing.md" in docs
    assert "## Golden Paths" in golden_paths
    assert "Minimal passive propagation" in golden_paths
    assert "Closed-loop rendezvous" in golden_paths
    assert "Mission recovery" in golden_paths
    assert "agent-capability-routing.md" in root_agents
    assert "agent-capability-routing.md" in public_agents
    assert "agent-capability-routing.md" in docs
    assert "Agent Task Cards" in task_cards
    assert "Passive Propagation" in task_cards
    assert "Agent Capability Routing" in capability_routing
    assert "Routing Table" in capability_routing
    assert "Public Core Boundary" in capability_routing
    assert "Evidence By Workflow" in capability_routing
    assert "Do not claim" in capability_routing
    assert "SGP4/general-perturbations propagation" in capability_routing
    assert "Statistical robustness from one or two deterministic runs" in capability_routing
    assert "not included in the public core" in root_agents

    public_agent_docs = {
        "AGENTS.md": root_agents,
        "agents/README.md": agents_readme,
        "agents/public/AGENTS.md": public_agents,
        "agents/public/evaluation-rubric.md": rubric,
        "docs/oel-agents.md": docs,
        "docs/agent-evaluation-packet.md": eval_packet,
        "docs/agent-capability-routing.md": capability_routing,
        "docs/agent-review-queries.md": review_queries,
        "docs/agent-feedback-loop.md": feedback_loop,
        "docs/agent-task-cards.md": task_cards,
    }
    assert "v1-local-workflows.md" in root_agents
    assert "v1-local-workflows.md" in public_agents
    assert "v1-local-workflows.md" in docs
    assert "does not upsell Pro" in rubric
    assert "campaign-scale controller ranking" in eval_packet
    assert "private Hosted/qualification" in eval_packet
    for path, text in public_agent_docs.items():
        assert "agents/pro" not in text, path
        assert "sim.pro_" not in text, path


def test_public_agent_docs_support_natural_user_requests() -> None:
    public_agents = (ROOT / "agents" / "public" / "AGENTS.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "oel-agents.md").read_text(encoding="utf-8")

    assert not (ROOT / "agents" / "examples" / "prompts").exists()
    for text in (public_agents, docs):
        assert "ordinary language" in text or "plain language" in text
        assert "Do not require" in text or "should not need special wording" in text
        assert "validate before running" in text or "Validate the config" in text
        assert "Make me a simple satellite propagation scenario" in text


def test_public_agent_task_cards_define_checked_workflows() -> None:
    assert {path.stem for path in AGENT_TASK_CARDS} == {
        "attitude_hold",
        "closed_loop_rendezvous",
        "compare_one_change",
        "completed_run_continuation",
        "coverage_and_directed_link",
        "ground_access_from_tle",
        "mission_reconstitution_trade_space",
        "mission_recovery_plus_c_burn",
        "ogp_sgp4_propagation",
        "ogp_sdp4_propagation",
        "passive_propagation",
        "python_api_minimal_propagation",
    }

    for card_path in AGENT_TASK_CARDS:
        text = card_path.read_text(encoding="utf-8")
        example_path = _task_card_example_path(card_path)
        answer_path = _task_card_answer_example_path(card_path)
        sql_blocks = _task_card_sql_blocks(card_path)

        assert example_path.is_file(), card_path
        assert answer_path.is_file(), card_path
        assert "## User Prompt" in text
        assert "## Expected Agent Assumptions" in text
        assert "## Commands" in text
        assert "## Required Review Queries" in text
        assert "## Expected Answer Shape" in text
        assert "## Pass Criteria" in text
        assert "## Red Flags" in text
        assert sql_blocks, card_path
        for query in sql_blocks:
            assert query.upper().startswith(("SELECT", "WITH")), (card_path, query)

        answer = answer_path.read_text(encoding="utf-8")
        for section in ANSWER_EXAMPLE_REQUIRED_SECTIONS:
            assert section in answer, answer_path


def test_public_agent_golden_paths_reference_real_fixtures_and_saved_queries() -> None:
    text = (ROOT / "docs" / "agent-capability-routing.md").read_text(encoding="utf-8")

    for path_id, spec in GOLDEN_PATHS.items():
        for config_ref in spec["configs"]:
            assert config_ref in text, (path_id, config_ref)
            config_path = ROOT / config_ref
            assert config_path.is_file(), config_ref
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            assert config["outputs"]["review"]["enabled"] is True, config_ref
            assert str(config["outputs"]["output_dir"]).startswith("outputs/agents/"), config_ref

        for query_name in spec["queries"]:
            assert f"`{query_name}`" in text or f"--saved-query {query_name}" in text, (path_id, query_name)
            assert query_name in SAVED_REVIEW_QUERIES, (path_id, query_name)
            assert SAVED_REVIEW_QUERIES[query_name].sql.upper().startswith(("SELECT", "WITH"))


@pytest.mark.parametrize("example_path", AGENT_EXAMPLES, ids=AGENT_EXAMPLE_IDS)
def test_public_agent_generated_examples_validate_with_cli(example_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--config", str(example_path), "--validate-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert example_path.stem in proc.stdout
    assert "OK" in proc.stdout


@pytest.mark.parametrize("example_path", AGENT_EXAMPLES, ids=AGENT_EXAMPLE_IDS)
def test_public_agent_generated_examples_run_headlessly(example_path: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    outdir = tmp_path / example_path.stem
    config["outputs"]["output_dir"] = str(outdir)
    assert config["outputs"]["review"]["enabled"] is True

    cfg_path = tmp_path / example_path.name
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_simulation_config_file(cfg_path)

    assert result["scenario_name"] == config["scenario_name"]
    assert (outdir / "index.md").is_file()
    assert (outdir / "master_run_summary.json").is_file()
    assert (outdir / "review" / "run.sqlite").is_file()
    assert (outdir / "review" / "schema.json").is_file()
    assert not any(outdir.glob("*.png"))

    with sqlite3.connect(outdir / "review" / "run.sqlite") as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        metadata = conn.execute(
            "SELECT scenario_name, duration_s, samples FROM run_metadata"
        ).fetchone()

    assert {"run_metadata", "objects", "object_state", "metrics", "artifacts"}.issubset(table_names)
    assert metadata is not None
    assert metadata[0] == config["scenario_name"]

    summary = json.loads((outdir / "master_run_summary.json").read_text(encoding="utf-8"))
    assert summary["scenario_name"] == config["scenario_name"]
    assert summary["duration_s"] == config["simulator"]["duration_s"]
    assert sorted(summary["objects"]) == sorted(
        object_id for object_id, object_cfg in config["objects"].items() if object_cfg.get("enabled", True)
    )


@pytest.mark.parametrize("card_path", AGENT_TASK_CARDS, ids=AGENT_TASK_CARD_IDS)
def test_public_agent_task_card_review_queries_execute(card_path: Path, tmp_path: Path) -> None:
    example_path = _task_card_example_path(card_path)
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    outdir = tmp_path / card_path.stem
    config["outputs"]["output_dir"] = str(outdir)

    cfg_path = tmp_path / f"{card_path.stem}.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_simulation_config_file(cfg_path)

    db_path = outdir / "review" / "run.sqlite"
    assert db_path.is_file()

    with sqlite3.connect(db_path) as conn:
        for query in _task_card_sql_blocks(card_path):
            cursor = conn.execute(query)
            cursor.fetchall()
            assert cursor.description is not None, (card_path, query)


@pytest.mark.parametrize("example_path", AGENT_EXAMPLES, ids=AGENT_EXAMPLE_IDS)
def test_public_agent_saved_review_queries_execute(example_path: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    outdir = tmp_path / f"saved_query_{example_path.stem}"
    config["outputs"]["output_dir"] = str(outdir)
    cfg_path = tmp_path / example_path.name
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_simulation_config_file(cfg_path)

    query_names_by_example = {
        "public_agent_python_api_minimal_propagation": ["run_metadata", "objects", "artifacts", "passive_final_state"],
        "public_agent_single_satellite": ["run_metadata", "objects", "artifacts", "passive_final_state"],
        "public_agent_ogp_sgp4_propagation": [
            "run_metadata",
            "objects",
            "ogp_propagation_contract",
            "passive_final_state",
            "artifacts",
        ],
        "public_agent_ogp_sdp4_propagation": [
            "run_metadata",
            "objects",
            "ogp_propagation_contract",
            "object_final_state",
            "object_eci_radius_extrema",
            "artifacts",
        ],
        "public_agent_rendezvous_lqr": [
            "run_metadata",
            "rendezvous_metrics",
            "rendezvous_closest_approach",
            "relative_final_state",
            "burn_activity",
            "burn_events",
        ],
        "public_agent_mission_recovery_plus_c_burn": [
            "run_metadata",
            "burn_activity",
            "mission_recovery_summary",
            "mission_recovery_elements",
            "mission_recovery_candidates",
            "mission_recovery_burns",
        ],
        "public_agent_mission_reconstitution_trade_space": [
            "run_metadata",
            "burn_activity",
            "mission_recovery_summary",
            "mission_recovery_elements",
            "mission_recovery_candidates",
            "mission_recovery_burns",
        ],
        "public_agent_ground_access": ["run_metadata", "ground_access_summary", "ground_access_no_access_reasons"],
        "public_agent_attitude_hold": [
            "run_metadata",
            "attitude_rates_first_last",
            "attitude_state_first_last",
            "burn_activity",
        ],
    }
    with sqlite3.connect(outdir / "review" / "run.sqlite") as conn:
        for query_name in query_names_by_example[example_path.stem]:
            cursor = conn.execute(SAVED_REVIEW_QUERIES[query_name].sql)
            rows = cursor.fetchall()
            assert cursor.description is not None
            if not SAVED_REVIEW_QUERIES[query_name].allow_empty:
                assert rows, (example_path.stem, query_name)


def test_public_agent_saved_review_query_cli_lists_queries() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "sim.review", "--list-saved-queries"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "rendezvous_metrics:" in proc.stdout
    assert "ground_access_summary:" in proc.stdout
    assert "attitude_state_first_last:" in proc.stdout
    assert "[supported; tables=" in proc.stdout


def test_public_agent_relative_state_shape_fails_validate_only(tmp_path: Path) -> None:
    example_path = ROOT / "agents" / "examples" / "public_agent_rendezvous_lqr.yaml"
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    chaser_initial = config["objects"]["chaser"]["initial_state"]
    rel_state = chaser_initial["relative_to_target_ric"].pop("state")
    chaser_initial["state"] = rel_state
    config["outputs"]["output_dir"] = str(tmp_path / "bad_relative_state")
    cfg_path = tmp_path / "bad_relative_state.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--config", str(cfg_path), "--validate-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    diagnostic = proc.stdout + proc.stderr
    assert "relative_to_target_ric.state" in diagnostic
    assert "length-6 finite numeric list" in diagnostic


def test_public_agent_saved_review_query_cli_runs_query(tmp_path: Path) -> None:
    example_path = ROOT / "agents" / "examples" / "public_agent_single_satellite.yaml"
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    outdir = tmp_path / "saved_query_cli"
    config["outputs"]["output_dir"] = str(outdir)
    cfg_path = tmp_path / example_path.name
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_simulation_config_file(cfg_path)

    proc = subprocess.run(
        [sys.executable, "-m", "sim.review", str(outdir), "--saved-query", "run_metadata", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["rows"][0]["scenario_name"] == config["scenario_name"]


def test_prepare_agent_feedback_builds_public_safe_draft() -> None:
    draft = build_agent_feedback_issue(
        agent_tool="Codex",
        workflow_stage="Review-store query",
        user_goal="Summarize a public rendezvous run.",
        issue_summary="The agent could not find a saved query for the needed metric.",
        expected="The agent should be able to query the metric directly.",
        evidence=".venv/bin/python -m sim.review outputs/example --saved-query rendezvous_metrics",
        suggestion="Add or document the saved query.",
    )

    assert "# Agent Feedback Draft" in draft
    assert "Codex" in draft
    assert "Review-store query" in draft
    assert "The user approved submitting this public feedback." in draft
    assert "Do not include secrets" in draft
