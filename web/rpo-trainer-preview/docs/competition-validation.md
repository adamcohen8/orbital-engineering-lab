# Browser Arcade Competition Validation

This document captures the hosted Pursuit Arcade leaderboard validation path
without running the downloadable Python/OEL engine in the cloud.

## Scope

The browser competition path is a browser-native implementation aligned with
the checked-in local Pursuit Arcade scenario. Pursuit Arcade is web-only in the
level selector and intentionally excluded from the downloadable launcher. The
browser does not execute Python `run_game.py`:

- The web arcade owns a deterministic, browser-native implementation of the
  Pursuit Arcade competition model.
- The downloadable OEL trainer remains the canonical high-fidelity/local tool.
- Hosted leaderboard validation replays inputs instead of trusting
  client-submitted states or scores.

The current browser-native challenge is contract-checked against
`sim/game/configs/game_training_rpo_arcade_pursuit.yaml` for the first-round
setup and core scoring contract:

- scenario id: `rpo_arcade_pursuit`
- two-body Earth gravity
- 1 s fixed step
- 12,000 s initial time budget
- 3.0 m/s chaser delta-v budget
- 100 m range goal
- 0.1 m/s target defensive delta-v budget
- target defensive acceleration: 7.5e-6 km/s^2
- target defense activates on range OR closing-speed trigger
- desktop-style score: seconds remaining plus chaser/target delta-v remaining
  in mm/s, multiplied by difficulty

The replay-versioned browser challenge retains `mu = 398600.4418 km^3/s^2`,
while the current OEL engine constant is `398600.4415 km^3/s^2`. This bounded
exception preserves existing `web-two-body-v2` hashes and replay packets; the
Level 0 OEL-reference trajectory gate in `tests/preview-contract.test.mjs`
checks the practical browser-model continuity.

Implemented arcade-round parity:

- Multi-round reset/continuation after a cleared round.
- Round score weighted by round number.
- Remaining-time update with 75% target-orbital-period bonus plus
  unused-chaser-delta-v bonus at 1000 seconds per unused m/s.
- Goal tightening by 5 m per cleared round, down to a 5 m floor.
- Round 2+ randomized initial states with energy-matched in-track velocity.
- Every fifth boss round with randomized elliptical target true anomaly.
- Boss round score multiplier, 5000 second flat bonus, and bonus time
  transition.
- Target defensive delta-v budget ramp after round 20.
- New rounds restart at 1x speed and full-trajectory camera framing.
- Round transitions and replay summaries carry exact pass-tick clear range and
  speed values, while web RI/RC goal rings remain stable projected range
  references.
- Tschauner-Hempel-style elliptic linear coast projection for boss-round
  trajectory previews.
- Speed-dependent game tick helper matching the downloadable trainer policy
  while clamping to the browser challenge's canonical base step.

Remaining operational/parity work:

- Admin moderation for hiding attempts and resolving username disputes.
- Monthly challenge rotation and operational tooling.
- Stored high-score replay/debrief browsing beyond the static RI/RC plot
  artifacts.

The current implementation lives in:

- `src/competition/arcade-engine.js`
- `tools/validate-attempt.mjs`
- `fixtures/sample-valid-attempt.json`
- `tests/competition-engine.test.mjs`
- `supabase/schema.sql`

## Physics Model

The competition engine propagates the target and chaser as two inertial ECI
states under central Earth gravity. Player controls are expressed in the target
RIC frame, mapped into ECI acceleration, and integrated with fixed-step RK4.

This supports both circular and elliptical target orbits without relying on HCW
as the authoritative arcade propagator.

The implemented state flow is:

1. Convert target classical orbital elements to ECI.
2. Place the chaser from an initial target-RIC relative state.
3. On each simulation tick, apply recorded input events.
4. Map the active RIC command into ECI acceleration.
5. Apply deterministic seeded target defense pulses when enabled.
6. Propagate target/chaser with fixed-step two-body dynamics.
7. Recompute relative RIC state, delta-v, pass/fail, and score.

## Attempt Packets

The validator accepts standalone single-round packets shaped like:

```json
{
  "schema_version": 1,
  "challenge_id": "rpo_arcade_pursuit",
  "username": "ORBITACE",
  "email": "optional@example.edu",
  "client_build_hash": "web-build-id",
  "physics_version": "web-two-body-v2",
  "scoring_version": "pursuit-v1",
  "config_hash": "8f4352c7",
  "seed": 4242,
  "final_tick": 1200,
  "input_events": [
    { "tick": 5, "control": "iPlus", "state": "down" },
    { "tick": 45, "control": "iPlus", "state": "up" }
  ],
  "claimed_score": 0,
  "claimed_metrics": {
    "elapsed_s": 1200,
    "player_delta_v_m_s": 1.35
  }
}
```

Inputs are recorded by simulation tick, not wall-clock time. That is the main
determinism contract. `final_tick` tells the validator where the player ended
the attempt; omitted values mean replay through the full challenge time budget.

Full arcade runs use `schema_version: 2`, `attempt_type: "arcade_run"`, and a
`round_attempts` array. Each round attempt stores the `round_index`,
`final_tick`, and that round's recorded `input_events`. Validation replays the
rounds contiguously from round 1, recomputes score/time/bonus transitions, and
rejects claimed scores or metrics that do not match the canonical replay.

## Validation

The validator:

1. Checks challenge id, physics version, scoring version, and config hash.
2. Checks input events are sorted, in bounds, and use known controls.
3. Replays the attempt from the canonical challenge config and seed.
4. Recomputes trajectory, delta-v, pass/fail, metrics, and score.
5. Compares the recomputed result against the claimed result.
6. Returns `valid`, `invalid`, or `suspicious`.

Run locally with:

```bash
node web/rpo-trainer-preview/tools/validate-attempt.mjs \
  --attempt web/rpo-trainer-preview/fixtures/sample-valid-attempt.json
```

If `node` is not on your PATH, install Node.js or run the command through the
bundled Codex Node runtime while working in Codex.

To generate static plots from a validated replay:

```bash
node web/rpo-trainer-preview/tools/validate-attempt.mjs \
  --attempt web/rpo-trainer-preview/fixtures/sample-valid-attempt.json \
  --plot-dir outputs/web-competition-smoke
```

That writes:

- `validated-ri.svg`
- `validated-rc.svg`

The plot SVGs are generated from the recomputed server-side replay, not from
client-submitted images.

## Test

```bash
cd web/rpo-trainer-preview
npm test
```

The tests cover:

- stable canonical config hashing,
- speed-dependent game tick policy,
- RIC/ECI conversion round trip,
- deterministic replay,
- valid attempt acceptance,
- tampered score rejection,
- full multi-round arcade attempt validation,
- fixture validation,
- static SVG plot generation with burn markers.
- downloadable-YAML parity for Level 0 copy/budgets and Pursuit Arcade
  constants,
- browser HCW path comparison against OEL-generated Level 0 two-body
  references.

## Hosted Pieces

These require accounts/credentials and are not committed here:

1. Static hosting for `web/rpo-trainer-preview`.
2. A database project, such as Supabase.
3. Private tables plus the denormalized public leaderboard table from
   `supabase/schema.sql`.
4. Optional email provider credentials for proof-of-ownership links.
5. Admin controls for creating monthly challenges and hiding attempts.

## Completed Local Pieces

1. Pursuit Arcade is available from the web level selector.
2. Browser gameplay is wired to `src/competition/arcade-engine.js`.
3. The interactive session records `down`/`up` inputs by simulation tick.
4. Local debrief validation replays the attempt packet and reports validation
   status.
5. Multi-round arcade play and boss rounds are available locally.
6. Multi-round arcade attempt packets replay through the deterministic
   validator, including score, time, and round transition checks.
7. The hosted API validates submissions before inserting attempts, stores
   canonical plot SVGs, maintains denormalized public leaderboard rows, and
   supports verified email username locking.

## Suggested Next Implementation Steps

1. Add admin tooling to hide attempts and resolve username disputes.
2. Add monthly challenge rotation.
3. Add a replay/debrief viewer for stored high-score plots.
