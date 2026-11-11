"""Independent admission control and durable at-most-once command ledger."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from .contracts import PointingAction, Policy, digest
from .engineering import OelForecast, SpacecraftPackage


class CommandGate:
    """Owned by the endpoint. Callers never supply policy, models or evaluation results.

    A process crash after intent is recorded leaves an indeterminate request for
    reconciliation. It must never turn into an automatic second actuation.
    The endpoint serializes these methods with its plant lock.
    """

    def __init__(
        self,
        package: SpacecraftPackage,
        policy: Policy,
        observe,
        ledger: Path,
        *,
        wall_time=time.time,
        monotonic=time.monotonic,
        evaluator=None,
    ):
        self.package, self.policy, self.observe = package, policy, observe
        self.forecast = evaluator if evaluator is not None else OelForecast(package, policy)
        self.wall_time, self.monotonic = wall_time, monotonic
        self.tickets = {}
        ledger.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(ledger, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, action TEXT NOT NULL, result TEXT NOT NULL)"
        )
        self.db.commit()

    def prior(self, action: PointingAction):
        row = self.db.execute("SELECT action,result FROM requests WHERE id=?", (action.request_id,)).fetchone()
        if row:
            if row[0] != digest(action.to_dict()):
                return {"disposition": "rejected", "reason": "request ID reused with different action"}
            return {"disposition": "duplicate", "prior": json.loads(row[1]), "reason": "no second actuation"}
        return None

    def freshness(self, observation):
        try:
            timestamp = datetime.fromisoformat(observation["generation_time"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("generation time must include its UTC offset")
            generated = timestamp.timestamp()
            age = self.wall_time() - generated
            if age < -0.5 or age > self.policy.max_telemetry_age_s:
                return {"disposition": "insufficient_evidence", "reason": "stale or future telemetry", "age_s": age}
        except (KeyError, ValueError, TypeError):
            return {"disposition": "insufficient_evidence", "reason": "invalid telemetry timestamp"}
        return None

    def evaluate(self, action: PointingAction):
        prior = self.prior(action)
        if prior:
            return prior
        observation = self.observe()
        stale = self.freshness(observation)
        if stale:
            return stale
        result = self.forecast.evaluate(action, observation)
        if result["disposition"] == "permitted":
            now = self.monotonic()
            self.tickets = {k: v for k, v in self.tickets.items() if v["expires"] > now}
            if len(self.tickets) >= 256:
                return {"disposition": "rejected", "reason": "approval capacity exhausted"}
            ticket = secrets.randbelow(2**32 - 1) + 1
            while ticket in self.tickets:
                ticket = secrets.randbelow(2**32 - 1) + 1
            self.tickets[ticket] = {
                "action": digest(action.to_dict()),
                "observation": digest(observation),
                "model": self.package.identity,
                "policy": self.policy.identity,
                "expires": now + self.policy.ticket_ttl_s,
            }
            result = {**result, "ticket": ticket, "expires_in_s": self.policy.ticket_ttl_s}
        return result

    def admit(self, action: PointingAction, ticket: int):
        prior = self.prior(action)
        if prior:
            return prior
        approval = self.tickets.pop(ticket, None)
        if not approval:
            return {"disposition": "rejected", "reason": "unknown or consumed approval"}
        if approval["action"] != digest(action.to_dict()):
            return {"disposition": "rejected", "reason": "command differs from evaluated action"}
        if approval["model"] != self.package.identity or approval["policy"] != self.policy.identity:
            return {"disposition": "rejected", "reason": "model or policy changed"}
        if approval["expires"] <= self.monotonic():
            return {"disposition": "rejected", "reason": "expired approval"}
        observation = self.observe()
        stale = self.freshness(observation)
        if stale:
            return stale
        if approval["observation"] != digest(observation):
            return {"disposition": "rejected", "reason": "telemetry changed; reevaluation required"}
        # Recompute; an agent cannot substitute its own forecast evidence.
        result = self.forecast.evaluate(action, observation)
        if result["disposition"] == "permitted":
            self.record(
                action, {"status": "indeterminate", "reason": "actuation intent recorded; reconcile after crash"}
            )
        return result

    def record(self, action, result):
        self.db.execute(
            "INSERT OR REPLACE INTO requests VALUES (?,?,?)",
            (action.request_id, digest(action.to_dict()), json.dumps(result)),
        )
        self.db.commit()

    def close(self):
        self.db.close()
