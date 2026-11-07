import { test } from "node:test";
import assert from "node:assert/strict";

import { buildChallengeRecord, validateAttemptPacket } from "../src/competition/arcade-engine.js";

test("standalone validator rejects non-numeric final_tick instead of accepting an empty replay", () => {
  const record = buildChallengeRecord();
  const attempt = {
    schema_version: 1,
    challenge_id: record.challenge_id,
    username: "AUDIT",
    physics_version: record.physics_version,
    scoring_version: record.scoring_version,
    config_hash: record.config_hash,
    seed: 1,
    final_tick: "not-a-tick",
    input_events: [],
    claimed_score: 0,
    claimed_metrics: {},
  };

  const result = validateAttemptPacket(attempt, record);
  assert.equal(result.status, "invalid");
  assert.match(result.errors.join("\n"), /Invalid final_tick/);
});
