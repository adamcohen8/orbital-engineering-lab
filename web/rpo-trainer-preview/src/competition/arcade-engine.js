const DEFAULT_EPSILON = 1.0e-12;
const HASH_OFFSET = 2166136261;
const HASH_PRIME = 16777619;

export const CONTROL_IDS = Object.freeze(["rPlus", "rMinus", "iPlus", "iMinus", "cPlus", "cMinus"]);
export const DEFAULT_SPEED_DT_SCHEDULE = Object.freeze([
  [10.0, 2.0],
  [25.0, 2.0],
  [50.0, 5.0],
  [100.0, 10.0],
]);

export const DEFAULT_PURSUIT_CHALLENGE = Object.freeze({
  challenge_id: "rpo_arcade_pursuit",
  title: "Pursuit Arcade",
  physics_version: "web-two-body-v2",
  scoring_version: "pursuit-v1",
  mu_km3_s2: 398600.4418,
  dt_s: 1.0,
  max_time_s: 12000.0,
  max_player_accel_km_s2: 1.5e-5,
  max_delta_v_m_s: 3.0,
  max_target_delta_v_m_s: 0.1,
  goal_range_km: 0.1,
  goal_speed_km_s: null,
  difficulty: "easy",
  target_coes: {
    a_km: 7000.0,
    ecc: 0.0,
    inc_deg: 45.0,
    raan_deg: 0.0,
    argp_deg: 0.0,
    true_anomaly_deg: 0.0,
  },
  chaser_initial_ric: {
    r_km: 0.0,
    i_km: -1.8,
    c_km: 0.0,
    rd_km_s: 0.0,
    id_km_s: 0.0,
    cd_km_s: 0.0,
  },
  target_defense: {
    enabled: true,
    trigger_range_km: 1.2,
    trigger_closing_speed_km_s: 0.00025,
    keepout_radius_km: 0.1,
    max_accel_km_s2: 7.5e-6,
    max_delta_v_m_s: 0.1,
    delta_v_ramp_after_round: 20,
    delta_v_ramp_step_m_s: 0.01,
    pulse_period_s: 120.0,
    cross_track_bias: 0.65,
  },
  arcade: {
    initial_time_s: 12000.0,
    round_bonus_time_s: 0.0,
    delta_v_bonus_time_per_m_s: 1000.0,
    goal_range_step_km: 0.005,
    min_goal_range_km: 0.005,
    boss_round_interval: 5,
    boss: {
      eccentricity_start: 0.05,
      eccentricity_step: 0.05,
      eccentricity_max: 0.20,
      target_coes: {
        a_km: 9000.0,
        ecc: 0.05,
        inc_deg: 45.0,
        raan_deg: 0.0,
        argp_deg: 0.0,
        true_anomaly_deg: 0.0,
      },
      true_anomaly_range_deg: [0.0, 360.0],
      coast_prediction_model: "tschauner_hempel",
      music_track: "28_high_shred_boss_riff.wav",
      score_multiplier: 2.0,
      bonus_time_s: 5000.0,
    },
    random_initial_state: {
      enabled: true,
      radial_range_km: [-1.0, 1.0],
      in_track_range_km: [-10.0, 10.0],
      cross_track_range_km: [-1.0, 1.0],
      min_range_km: 5.0,
      cross_track_rate_range_km_s: [-0.001, 0.001],
    },
  },
  scoring: {
    difficulty_multiplier: 1,
  },
});

export function gameTickDtS({ baseDtS, speedMultiple, speedDtSchedule = DEFAULT_SPEED_DT_SCHEDULE } = {}) {
  const base = positiveNumber(baseDtS, "baseDtS");
  const speed = positiveNumber(speedMultiple ?? 1.0, "speedMultiple");
  const schedule = Array.isArray(speedDtSchedule) ? speedDtSchedule : DEFAULT_SPEED_DT_SCHEDULE;
  let chosen = base;
  for (const row of schedule) {
    const thresholdSpeed = Number(Array.isArray(row) ? row[0] : row?.speed);
    const dtS = Number(Array.isArray(row) ? row[1] : row?.dt_s);
    if (Number.isFinite(thresholdSpeed) && thresholdSpeed > 0 && Number.isFinite(dtS) && dtS > 0) {
      if (speed >= thresholdSpeed - 1.0e-9) chosen = dtS;
    }
  }
  return Math.max(Math.min(chosen, base), DEFAULT_EPSILON);
}

export function buildChallengeRecord(config = DEFAULT_PURSUIT_CHALLENGE) {
  const canonical = normalizeChallengeConfig(config);
  return {
    challenge_id: canonical.challenge_id,
    physics_version: canonical.physics_version,
    scoring_version: canonical.scoring_version,
    config_hash: hashCanonicalJson(canonical),
    config: canonical,
  };
}

export function normalizeChallengeConfig(config = DEFAULT_PURSUIT_CHALLENGE) {
  return deepFreeze(cloneCanonical({ ...DEFAULT_PURSUIT_CHALLENGE, ...config }));
}

export function makeAttemptPacket({
  challengeRecord,
  username,
  email = "",
  seed = 1,
  input_events = [],
  result = null,
  client_build_hash = "local-dev",
}) {
  const record = challengeRecord || buildChallengeRecord();
  const replay = result || runPursuitReplay(record.config, { seed, input_events });
  return {
    schema_version: 1,
    challenge_id: record.challenge_id,
    username: String(username || "anonymous").trim(),
    email: String(email || "").trim(),
    client_build_hash: String(client_build_hash || "local-dev"),
    physics_version: record.physics_version,
    scoring_version: record.scoring_version,
    config_hash: record.config_hash,
    seed: integerSeed(seed),
    final_tick: result?.ticks === undefined ? undefined : Math.floor(Number(result.ticks)),
    input_events: normalizeInputEvents(input_events),
    claimed_score: replay.score,
    claimed_metrics: replay.metrics,
  };
}

export function makeArcadeAttemptPacket({
  challengeRecord,
  username,
  email = "",
  seed = 1,
  round_attempts = [],
  result = null,
  client_build_hash = "local-dev",
}) {
  const record = challengeRecord || buildChallengeRecord();
  const replay = result || runPursuitArcadeReplay(record.config, { seed, round_attempts });
  return {
    schema_version: 2,
    attempt_type: "arcade_run",
    challenge_id: record.challenge_id,
    username: String(username || "anonymous").trim(),
    email: String(email || "").trim(),
    client_build_hash: String(client_build_hash || "local-dev"),
    physics_version: record.physics_version,
    scoring_version: record.scoring_version,
    config_hash: record.config_hash,
    seed: integerSeed(seed),
    round_attempts: normalizeRoundAttempts(round_attempts),
    claimed_score: replay.score,
    claimed_metrics: replay.metrics,
  };
}

export function createPursuitSession(config = DEFAULT_PURSUIT_CHALLENGE, options = {}) {
  const cfg = normalizeChallengeConfig(config);
  const seed = integerSeed(options.seed || Math.floor(Date.now() % 4294967296));
  const prng = mulberry32(seed);
  const initial = initialArcadeState(cfg);
  const activeControls = new Set();
  const pulseControls = new Set();
  const downTickByControl = new Map();
  const inputEvents = [];
  const history = [];
  const burnMarkers = [];
  let sim = {
    tick: 0,
    time_s: 0,
    target_reference: initial.target_reference,
    target: initial.target,
    chaser: initial.chaser,
    player_delta_v_m_s: 0,
    target_delta_v_m_s: 0,
    target_next_pulse_s: 0,
    target_pulse_remaining_s: 0,
    target_pulse_direction_ric: seededFixedDirection(prng),
  };
  let closestRangeKm = Infinity;
  let passed = false;
  let failed = false;
  let achievedTimeS = null;
  let achievedRangeKm = null;
  let achievedRelSpeedKmS = null;
  let terminalReason = "";

  const capture = () => {
    const rel = relativeRicState(sim.target, sim.chaser);
    const rangeKm = Math.hypot(rel.r_km, rel.i_km, rel.c_km);
    const relSpeedKmS = Math.hypot(rel.rd_km_s, rel.id_km_s, rel.cd_km_s);
    closestRangeKm = Math.min(closestRangeKm, rangeKm);
    const speedOk = cfg.goal_speed_km_s === null || cfg.goal_speed_km_s === undefined || relSpeedKmS <= cfg.goal_speed_km_s;
    if (!passed && !failed && rangeKm <= cfg.goal_range_km && speedOk) {
      passed = true;
      achievedTimeS = sim.time_s;
      achievedRangeKm = rangeKm;
      achievedRelSpeedKmS = relSpeedKmS;
      terminalReason = "Goal reached under the speed limit.";
    }
    if (!passed && !failed && sim.player_delta_v_m_s > cfg.max_delta_v_m_s + 1.0e-9) {
      failed = true;
      terminalReason = "Delta-v budget exhausted.";
    }
    if (!passed && !failed && sim.time_s >= cfg.max_time_s - 1.0e-9) {
      failed = true;
      terminalReason = "Time expired.";
    }
    const sample = historySample(sim, rel, activeControls, pulseControls);
    const last = history[history.length - 1];
    if (!last || last.tick !== sample.tick) history.push(sample);
    return { rel, rangeKm, relSpeedKmS };
  };

  capture();

  return {
    config: cfg,
    seed,
    setControl(control, active) {
      if (!CONTROL_IDS.includes(control)) throw new Error(`Unknown control id: ${control}.`);
      const shouldBeActive = Boolean(active);
      const isActive = activeControls.has(control);
      if (shouldBeActive === isActive) return;
      if (shouldBeActive) {
        activeControls.add(control);
        downTickByControl.set(control, sim.tick);
      } else {
        if (downTickByControl.get(control) === sim.tick) pulseControls.add(control);
        activeControls.delete(control);
        downTickByControl.delete(control);
      }
      inputEvents.push({ tick: sim.tick, control, state: shouldBeActive ? "down" : "up" });
    },
    setControls(controls) {
      const clamped = clampControls(controls || {});
      this.setControl("rPlus", clamped.r > 0.5);
      this.setControl("rMinus", clamped.r < -0.5);
      this.setControl("iPlus", clamped.i > 0.5);
      this.setControl("iMinus", clamped.i < -0.5);
      this.setControl("cPlus", clamped.c > 0.5);
      this.setControl("cMinus", clamped.c < -0.5);
    },
    step(count = 1) {
      const steps = Math.max(0, Math.floor(count));
      for (let idx = 0; idx < steps; idx += 1) {
        const current = capture();
        if (passed || failed) break;
        const controls = controlsFromActive(activeControls, pulseControls);
        if (Math.hypot(controls.r, controls.i, controls.c) > 0) {
          burnMarkers.push({
            tick: sim.tick,
            time_s: sim.time_s,
            controls: { ...controls },
            relative_ric: { ...current.rel },
          });
        }
        sim = stepArcadeState(sim, cfg, controls, cfg.dt_s, prng);
        pulseControls.clear();
      }
      capture();
      return this.snapshot();
    },
    snapshot() {
      const { rel, rangeKm, relSpeedKmS } = capture();
      const score = pursuitScore(cfg, {
        passed,
        failed,
        achieved_time_s: achievedTimeS,
        player_delta_v_m_s: sim.player_delta_v_m_s,
        target_delta_v_m_s: sim.target_delta_v_m_s,
      });
      return {
        tick: sim.tick,
        time_s: sim.time_s,
        relative_ric: {
          r_km: roundMetric(rel.r_km),
          i_km: roundMetric(rel.i_km),
          c_km: roundMetric(rel.c_km),
          rd_km_s: roundMetric(rel.rd_km_s),
          id_km_s: roundMetric(rel.id_km_s),
          cd_km_s: roundMetric(rel.cd_km_s),
        },
        range_km: roundMetric(rangeKm),
        relative_speed_km_s: roundMetric(relSpeedKmS),
        player_delta_v_m_s: roundMetric(sim.player_delta_v_m_s),
        target_delta_v_m_s: roundMetric(sim.target_delta_v_m_s),
        max_delta_v_m_s: roundMetric(cfg.max_delta_v_m_s),
        max_target_delta_v_m_s: roundMetric(cfg.max_target_delta_v_m_s ?? cfg.target_defense?.max_delta_v_m_s ?? 0),
        closest_range_km: roundMetric(closestRangeKm),
        achieved_range_km: achievedRangeKm === null ? null : roundMetric(achievedRangeKm),
        achieved_relative_speed_km_s: achievedRelSpeedKmS === null ? null : roundMetric(achievedRelSpeedKmS),
        passed,
        failed,
        terminal: passed || failed,
        terminal_reason: terminalReason,
        score,
        active_controls: controlsFromActive(activeControls, pulseControls),
        input_events: inputEvents.map((event) => ({ ...event })),
        target_reference_state_eci: eciStateBlock(sim.target_reference || sim.target),
        history: history.map((sample) => ({ ...sample, relative_ric: { ...sample.relative_ric } })),
      };
    },
    result() {
      const snap = this.snapshot();
      return {
        passed,
        failed,
        score: snap.score,
        seed,
        ticks: sim.tick,
        elapsed_s: sim.time_s,
        metrics: {
          achieved_time_s: achievedTimeS,
          achieved_range_km: achievedRangeKm === null ? null : roundMetric(achievedRangeKm),
          achieved_relative_speed_km_s: achievedRelSpeedKmS === null ? null : roundMetric(achievedRelSpeedKmS),
          elapsed_s: roundMetric(sim.time_s),
          closest_range_km: roundMetric(closestRangeKm),
          final_range_km: snap.range_km,
          final_relative_speed_km_s: snap.relative_speed_km_s,
          player_delta_v_m_s: roundMetric(sim.player_delta_v_m_s),
          target_delta_v_m_s: roundMetric(sim.target_delta_v_m_s),
        },
        burn_markers: compactBurnMarkers(burnMarkers),
        history: snap.history,
      };
    },
    attemptPacket({ challengeRecord, username = "LOCAL", email = "", client_build_hash = "local-preview" } = {}) {
      return makeAttemptPacket({
        challengeRecord: challengeRecord || buildChallengeRecord(cfg),
        username,
        email,
        seed,
        input_events: inputEvents,
        result: this.result(),
        client_build_hash,
      });
    },
  };
}

export function createPursuitArcadeSession(config = DEFAULT_PURSUIT_CHALLENGE, options = {}) {
  const baseConfig = normalizeChallengeConfig(config);
  const arcadeSeed = integerSeed(options.seed || Math.floor(Date.now() % 4294967296));
  let roundIndex = Math.max(1, Math.floor(Number(options.startRoundIndex ?? options.start_round_index ?? 1)));
  let totalScore = Math.max(0, Math.floor(Number(options.initialTotalScore ?? options.initial_total_score ?? 0)));
  let remainingTimeS = Number(options.remainingTimeS ?? options.remaining_time_s ?? baseConfig.arcade?.initial_time_s ?? baseConfig.max_time_s);
  let transition = null;
  let terminal = false;
  let terminalReason = "";
  let roundSummaries = [];
  let roundAttempts = [];
  let session = startRound();

  function startRound() {
    const roundConfig = arcadeRoundConfig(baseConfig, {
      arcadeSeed,
      roundIndex,
      remainingTimeS,
    });
    transition = null;
    return createPursuitSession(roundConfig, { seed: roundSeed(arcadeSeed, roundIndex, 977) });
  }

  function clearRoundIfNeeded() {
    const snap = session.snapshot();
    if (!snap.passed) {
      if (snap.failed) {
        terminal = true;
        terminalReason = snap.terminal_reason || "Arcade run ended.";
      }
      return;
    }
    const roundConfig = session.config;
    const result = session.result();
    const attempt = arcadeRoundAttemptFromSession(session, roundIndex);
    const roundScore = arcadeRoundWeightedScore(roundConfig, result, roundIndex);
    const timeUsedS = result.metrics.achieved_time_s ?? result.elapsed_s;
    const bonusTimeS = arcadeRoundTimeBonus(baseConfig, roundConfig, result, roundIndex);
    totalScore += roundScore;
    remainingTimeS = Math.max(remainingTimeS - timeUsedS, 0) + bonusTimeS;
    const clearedRound = roundIndex;
    roundSummaries.push({
      round_index: clearedRound,
      score: roundScore,
      total_score: totalScore,
      time_used_s: roundMetric(timeUsedS),
      bonus_time_s: roundMetric(bonusTimeS),
      remaining_time_s: roundMetric(remainingTimeS),
      boss: arcadeRoundIsBoss(baseConfig, clearedRound),
      goal_range_km: roundConfig.goal_range_km,
      clear_range_km:
        result.metrics.achieved_range_km ?? result.metrics.closest_range_km ?? result.metrics.final_range_km,
      clear_relative_speed_km_s: result.metrics.achieved_relative_speed_km_s ?? result.metrics.final_relative_speed_km_s,
    });
    roundAttempts.push(attempt);
    roundIndex += 1;
    transition = {
      cleared_round_index: clearedRound,
      next_round_index: roundIndex,
      round_score: roundScore,
      total_score: totalScore,
      time_used_s: roundMetric(timeUsedS),
      bonus_time_s: roundMetric(bonusTimeS),
      next_time_budget_s: roundMetric(remainingTimeS),
      next_goal_range_km: arcadeRoundGoalRange(baseConfig, roundIndex),
      next_is_boss: arcadeRoundIsBoss(baseConfig, roundIndex),
      goal_range_km: roundMetric(roundConfig.goal_range_km),
      clear_range_km: roundMetric(
        result.metrics.achieved_range_km ?? result.metrics.closest_range_km ?? result.metrics.final_range_km,
      ),
      clear_relative_speed_km_s: roundMetric(
        result.metrics.achieved_relative_speed_km_s ?? result.metrics.final_relative_speed_km_s,
      ),
    };
  }

  return {
    get config() {
      return session.config;
    },
    get seed() {
      return arcadeSeed;
    },
    setControl(control, active) {
      if (terminal || transition) return;
      session.setControl(control, active);
    },
    setControls(controls) {
      if (terminal || transition) return;
      session.setControls(controls);
    },
    step(count = 1) {
      const steps = Math.max(0, Math.floor(count));
      for (let idx = 0; idx < steps; idx += 1) {
        if (terminal || transition) break;
        session.step(1);
        clearRoundIfNeeded();
      }
      return this.snapshot();
    },
    continueNextRound() {
      if (terminal || !transition) return this.snapshot();
      session = startRound();
      return this.snapshot();
    },
    snapshot() {
      const snap = session.snapshot();
      const currentRoundScore = transition ? 0 : snap.score;
      return {
        ...snap,
        round_index: roundIndex,
        total_score: totalScore,
        score: totalScore + currentRoundScore,
        round_score: transition?.round_score ?? currentRoundScore,
        remaining_time_s: roundMetric(remainingTimeS),
        goal_range_km: session.config.goal_range_km,
        is_boss_round: arcadeRoundIsBoss(baseConfig, roundIndex),
        round_transition: transition,
        terminal: terminal || snap.failed,
        failed: terminal || snap.failed,
        passed: false,
        terminal_reason: terminalReason || snap.terminal_reason,
        round_summaries: roundSummaries.map((summary) => ({ ...summary })),
      };
    },
    result() {
      const snap = this.snapshot();
      const attempts = arcadeRoundAttemptsWithCurrent(roundAttempts, session, roundIndex);
      return {
        ...session.result(),
        passed: false,
        failed: Boolean(snap.failed),
        score: snap.score,
        seed: arcadeSeed,
        round_index: roundIndex,
        total_score: totalScore,
        remaining_time_s: snap.remaining_time_s,
        round_summaries: snap.round_summaries,
        round_attempts: attempts,
        metrics: arcadeRunMetrics(snap, attempts),
      };
    },
    attemptPacket({ challengeRecord, username = "LOCAL", email = "", client_build_hash = "local-preview" } = {}) {
      return makeArcadeAttemptPacket({
        challengeRecord: challengeRecord || buildChallengeRecord(baseConfig),
        username,
        email,
        seed: arcadeSeed,
        round_attempts: arcadeRoundAttemptsWithCurrent(roundAttempts, session, roundIndex),
        result: this.result(),
        client_build_hash,
      });
    },
  };
}

export function validateAttemptPacket(attempt, challengeRecord, options = {}) {
  if (attempt?.attempt_type === "arcade_run" || Array.isArray(attempt?.round_attempts)) {
    return validateArcadeAttemptPacket(attempt, challengeRecord, options);
  }
  const record = challengeRecord || buildChallengeRecord();
  const errors = [];
  const warnings = [];
  if (!attempt || typeof attempt !== "object") {
    return { status: "invalid", errors: ["Attempt packet must be an object."], warnings, replay: null };
  }
  if (attempt.challenge_id !== record.challenge_id) errors.push("challenge_id does not match the active challenge.");
  if (attempt.physics_version !== record.physics_version) errors.push("physics_version does not match the active challenge.");
  if (attempt.scoring_version !== record.scoring_version) errors.push("scoring_version does not match the active challenge.");
  if (attempt.config_hash !== record.config_hash) errors.push("config_hash does not match the canonical challenge config.");
  if (!String(attempt.username || "").trim()) errors.push("username is required.");
  if (attempt.final_tick !== undefined) {
    const finalTick = Number(attempt.final_tick);
    const maxTick = Math.ceil(Number(record.config.max_time_s) / Number(record.config.dt_s));
    if (
      typeof attempt.final_tick !== "number"
      || !Number.isSafeInteger(finalTick)
      || finalTick < 0
      || finalTick > maxTick
    ) {
      errors.push(`Invalid final_tick: ${attempt.final_tick}.`);
    }
  }

  let inputEvents = [];
  try {
    inputEvents = normalizeInputEvents(attempt.input_events || []);
    validateInputEvents(inputEvents, record.config);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
  if (inputEvents.length > (options.suspicious_input_event_count || 2000)) {
    warnings.push("Input event count is unusually high for this challenge.");
  }
  if (errors.length > 0) {
    return { status: "invalid", errors, warnings, replay: null };
  }

  const replay = runPursuitReplay(record.config, {
    seed: attempt.seed,
    input_events: inputEvents,
    final_tick: attempt.final_tick,
    sample_stride_ticks: options.sample_stride_ticks,
  });
  const metricErrors = compareClaimedResult(attempt, replay, options);
  errors.push(...metricErrors);
  const status = errors.length > 0 ? "invalid" : warnings.length > 0 ? "suspicious" : "valid";
  return {
    status,
    errors,
    warnings,
    replay,
    canonical_score: replay.score,
    canonical_metrics: replay.metrics,
  };
}

export function validateArcadeAttemptPacket(attempt, challengeRecord, options = {}) {
  const record = challengeRecord || buildChallengeRecord();
  const errors = [];
  const warnings = [];
  if (!attempt || typeof attempt !== "object") {
    return { status: "invalid", errors: ["Attempt packet must be an object."], warnings, replay: null };
  }
  if (attempt.challenge_id !== record.challenge_id) errors.push("challenge_id does not match the active challenge.");
  if (attempt.physics_version !== record.physics_version) errors.push("physics_version does not match the active challenge.");
  if (attempt.scoring_version !== record.scoring_version) errors.push("scoring_version does not match the active challenge.");
  if (attempt.config_hash !== record.config_hash) errors.push("config_hash does not match the canonical challenge config.");
  if (!String(attempt.username || "").trim()) errors.push("username is required.");

  let roundAttempts = [];
  try {
    roundAttempts = normalizeRoundAttempts(attempt.round_attempts || []);
    validateRoundAttempts(roundAttempts, record.config);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
  const eventCount = roundAttempts.reduce((total, round) => total + round.input_events.length, 0);
  if (eventCount > (options.suspicious_input_event_count || 5000)) {
    warnings.push("Input event count is unusually high for this arcade run.");
  }
  if (errors.length > 0) {
    return { status: "invalid", errors, warnings, replay: null };
  }

  let replay = null;
  try {
    replay = runPursuitArcadeReplay(record.config, {
      seed: attempt.seed,
      round_attempts: roundAttempts,
      sample_stride_ticks: options.sample_stride_ticks,
    });
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
  if (!replay) {
    return { status: "invalid", errors, warnings, replay: null };
  }
  errors.push(...compareClaimedArcadeResult(attempt, replay, options));
  const status = errors.length > 0 ? "invalid" : warnings.length > 0 ? "suspicious" : "valid";
  return {
    status,
    errors,
    warnings,
    replay,
    canonical_score: replay.score,
    canonical_metrics: replay.metrics,
  };
}

export function runPursuitReplay(config = DEFAULT_PURSUIT_CHALLENGE, replay = {}) {
  const cfg = normalizeChallengeConfig(config);
  const dtS = positiveNumber(cfg.dt_s, "dt_s");
  const challengeMaxTicks = Math.max(0, Math.ceil(positiveNumber(cfg.max_time_s, "max_time_s") / dtS));
  const replayStopTick = replay.final_tick === undefined ? challengeMaxTicks : Math.floor(Number(replay.final_tick));
  const maxTicks = Math.max(0, Math.min(challengeMaxTicks, replayStopTick));
  const inputEvents = normalizeInputEvents(replay.input_events || []);
  validateInputEvents(inputEvents, cfg);
  const eventMap = eventsByTick(inputEvents);
  const activeControls = new Set();
  const prng = mulberry32(integerSeed(replay.seed || 1));
  const initial = initialArcadeState(cfg);
  let sim = {
    tick: 0,
    time_s: 0,
    target_reference: initial.target_reference,
    target: initial.target,
    chaser: initial.chaser,
    player_delta_v_m_s: 0,
    target_delta_v_m_s: 0,
    target_next_pulse_s: 0,
    target_pulse_remaining_s: 0,
    target_pulse_direction_ric: seededFixedDirection(prng),
  };
  const sampleStrideTicks = Math.max(1, Math.floor(replay.sample_stride_ticks || 1));
  const history = [];
  let passed = false;
  let failed = false;
  let achievedTimeS = null;
  let achievedRangeKm = null;
  let achievedRelSpeedKmS = null;
  let closestRangeKm = Infinity;
  let finalRelativeSpeedKmS = Infinity;
  let burnMarkers = [];

  for (let tick = 0; tick <= maxTicks; tick += 1) {
    const pulseControls = applyTickEvents(activeControls, eventMap.get(tick) || []);
    const rel = relativeRicState(sim.target, sim.chaser);
    const rangeKm = Math.hypot(rel.r_km, rel.i_km, rel.c_km);
    const relSpeedKmS = Math.hypot(rel.rd_km_s, rel.id_km_s, rel.cd_km_s);
    closestRangeKm = Math.min(closestRangeKm, rangeKm);
    finalRelativeSpeedKmS = relSpeedKmS;
    if (tick % sampleStrideTicks === 0 || tick === maxTicks) {
      history.push(historySample(sim, rel, activeControls, pulseControls));
    }
    const speedOk = cfg.goal_speed_km_s === null || cfg.goal_speed_km_s === undefined || relSpeedKmS <= cfg.goal_speed_km_s;
    if (rangeKm <= cfg.goal_range_km && speedOk) {
      passed = true;
      achievedTimeS = sim.time_s;
      achievedRangeKm = rangeKm;
      achievedRelSpeedKmS = relSpeedKmS;
      break;
    }
    if (sim.player_delta_v_m_s > cfg.max_delta_v_m_s + 1.0e-9) {
      failed = true;
      break;
    }
    if (tick >= maxTicks) break;
    const controls = controlsFromActive(activeControls, pulseControls);
    const controlMagnitude = Math.hypot(controls.r, controls.i, controls.c);
    if (controlMagnitude > 0) {
      burnMarkers.push({ tick, time_s: sim.time_s, controls: { ...controls }, relative_ric: { ...rel } });
    }
    sim = stepArcadeState(sim, cfg, controls, dtS, prng);
  }

  const score = pursuitScore(cfg, {
    passed,
    failed,
    achieved_time_s: achievedTimeS,
    player_delta_v_m_s: sim.player_delta_v_m_s,
    target_delta_v_m_s: sim.target_delta_v_m_s,
  });
  return {
    passed,
    failed,
    score,
    seed: integerSeed(replay.seed || 1),
    ticks: sim.tick,
    elapsed_s: sim.time_s,
    metrics: {
      achieved_time_s: achievedTimeS,
      achieved_range_km: achievedRangeKm === null ? null : roundMetric(achievedRangeKm),
      achieved_relative_speed_km_s: achievedRelSpeedKmS === null ? null : roundMetric(achievedRelSpeedKmS),
      elapsed_s: sim.time_s,
      closest_range_km: roundMetric(closestRangeKm),
      final_range_km: roundMetric(rangeFromRelativeSample(history[history.length - 1])),
      final_relative_speed_km_s: roundMetric(finalRelativeSpeedKmS),
      player_delta_v_m_s: roundMetric(sim.player_delta_v_m_s),
      target_delta_v_m_s: roundMetric(sim.target_delta_v_m_s),
    },
    burn_markers: compactBurnMarkers(burnMarkers),
    history,
  };
}

export function runPursuitArcadeReplay(config = DEFAULT_PURSUIT_CHALLENGE, replay = {}) {
  const baseConfig = normalizeChallengeConfig(config);
  const arcadeSeed = integerSeed(replay.seed || 1);
  const roundAttempts = normalizeRoundAttempts(replay.round_attempts || []);
  validateRoundAttempts(roundAttempts, baseConfig);
  let totalScore = 0;
  let remainingTimeS = Number(baseConfig.arcade?.initial_time_s ?? baseConfig.max_time_s);
  let terminalReason = "";
  let failed = false;
  const roundSummaries = [];
  const history = [];
  const burnMarkers = [];
  let finalRoundReplay = null;
  let finalRoundIndex = 1;

  roundAttempts.forEach((attempt, idx) => {
    if (failed) return;
    const roundIndex = attempt.round_index;
    finalRoundIndex = roundIndex;
    if (roundIndex !== idx + 1) {
      failed = true;
      terminalReason = `Round attempts must be contiguous from round 1; found round ${roundIndex}.`;
      return;
    }
    const roundConfig = arcadeRoundConfig(baseConfig, {
      arcadeSeed,
      roundIndex,
      remainingTimeS,
    });
    const roundReplay = runPursuitReplay(roundConfig, {
      seed: roundSeed(arcadeSeed, roundIndex, 977),
      input_events: attempt.input_events,
      final_tick: attempt.final_tick,
      sample_stride_ticks: replay.sample_stride_ticks,
    });
    finalRoundReplay = roundReplay;
    history.push(...roundReplay.history.map((sample) => ({ ...sample, round_index: roundIndex })));
    burnMarkers.push(...roundReplay.burn_markers.map((marker) => ({ ...marker, round_index: roundIndex })));
    if (!roundReplay.passed) {
      failed = true;
      terminalReason = roundReplay.failed ? "Arcade run ended." : "Arcade run stopped before clearing the round.";
      return;
    }
    const roundScore = arcadeRoundWeightedScore(roundConfig, roundReplay, roundIndex);
    const timeUsedS = roundReplay.metrics.achieved_time_s ?? roundReplay.elapsed_s;
    const bonusTimeS = arcadeRoundTimeBonus(baseConfig, roundConfig, roundReplay, roundIndex);
    totalScore += roundScore;
    remainingTimeS = Math.max(remainingTimeS - timeUsedS, 0) + bonusTimeS;
    roundSummaries.push({
      round_index: roundIndex,
      score: roundScore,
      total_score: totalScore,
      time_used_s: roundMetric(timeUsedS),
      bonus_time_s: roundMetric(bonusTimeS),
      remaining_time_s: roundMetric(remainingTimeS),
      boss: arcadeRoundIsBoss(baseConfig, roundIndex),
      goal_range_km: roundConfig.goal_range_km,
      clear_range_km:
        roundReplay.metrics.achieved_range_km ?? roundReplay.metrics.closest_range_km ?? roundReplay.metrics.final_range_km,
      clear_relative_speed_km_s:
        roundReplay.metrics.achieved_relative_speed_km_s ?? roundReplay.metrics.final_relative_speed_km_s,
    });
  });

  const finalMetrics = finalRoundReplay?.metrics || {};
  return {
    passed: false,
    failed,
    terminal: failed,
    terminal_reason: terminalReason,
    score: totalScore,
    seed: arcadeSeed,
    round_index: finalRoundIndex,
    total_score: totalScore,
    remaining_time_s: roundMetric(remainingTimeS),
    round_summaries: roundSummaries,
    round_attempts: roundAttempts,
    metrics: {
      rounds_cleared: roundSummaries.length,
      final_round_index: finalRoundIndex,
      remaining_time_s: roundMetric(remainingTimeS),
      final_range_km: finalMetrics.final_range_km ?? null,
      final_relative_speed_km_s: finalMetrics.final_relative_speed_km_s ?? null,
      player_delta_v_m_s: finalMetrics.player_delta_v_m_s ?? null,
      target_delta_v_m_s: finalMetrics.target_delta_v_m_s ?? null,
    },
    burn_markers: burnMarkers,
    history,
  };
}

export function trajectoryPlotSvg(result, plane = "RI", options = {}) {
  const normalizedPlane = String(plane || "RI").toUpperCase();
  const xKey = normalizedPlane === "RC" ? "c_km" : "i_km";
  const yKey = "r_km";
  const width = Math.max(240, Number(options.width || 720));
  const height = Math.max(180, Number(options.height || 480));
  const margin = Number(options.margin || 42);
  const history = Array.isArray(result?.history) ? result.history : [];
  const points = history.map((sample) => sample.relative_ric || sample).filter((sample) => isFiniteNumber(sample[xKey]) && isFiniteNumber(sample[yKey]));
  const extents = plotExtents(points, xKey, yKey, options.min_span_km || 0.25);
  const toPx = (sample) => {
    const x = margin + ((sample[xKey] - extents.minX) / (extents.maxX - extents.minX)) * (width - margin * 2);
    const y = height - margin - ((sample[yKey] - extents.minY) / (extents.maxY - extents.minY)) * (height - margin * 2);
    return { x, y };
  };
  const path = points.map((point, idx) => {
    const px = toPx(point);
    return `${idx === 0 ? "M" : "L"} ${px.x.toFixed(2)} ${px.y.toFixed(2)}`;
  }).join(" ");
  const goalRadiusPx = (Number(options.goal_range_km || DEFAULT_PURSUIT_CHALLENGE.goal_range_km) / (extents.maxX - extents.minX)) * (width - margin * 2);
  const start = points[0] ? toPx(points[0]) : { x: width / 2, y: height / 2 };
  const end = points[points.length - 1] ? toPx(points[points.length - 1]) : start;
  const burnMarkers = (result?.burn_markers || [])
    .map((marker) => marker.relative_ric)
    .filter(Boolean)
    .filter((sample) => isFiniteNumber(sample[xKey]) && isFiniteNumber(sample[yKey]))
    .slice(0, 160)
    .map((sample) => toPx(sample));
  const title = escapeXml(`${normalizedPlane} Plane`);
  const score = escapeXml(String(result?.score ?? ""));
  const elapsed = escapeXml(formatSeconds(result?.metrics?.elapsed_s ?? result?.elapsed_s ?? 0));
  const deltaV = escapeXml(formatMps(result?.metrics?.player_delta_v_m_s ?? 0));
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${title} validated trajectory plot">
  <rect width="100%" height="100%" fill="#07111f"/>
  <text x="${margin}" y="26" fill="#e7eefc" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="700">${title}</text>
  <text x="${width - margin}" y="26" fill="#9fb0c7" font-family="Inter, Arial, sans-serif" font-size="12" text-anchor="end">Score ${score} | ${elapsed} | dV ${deltaV}</text>
  <line x1="${margin}" y1="${height / 2}" x2="${width - margin}" y2="${height / 2}" stroke="#26364d" stroke-width="1"/>
  <line x1="${width / 2}" y1="${margin}" x2="${width / 2}" y2="${height - margin}" stroke="#26364d" stroke-width="1"/>
  <circle cx="${toPx({ [xKey]: 0, [yKey]: 0 }).x.toFixed(2)}" cy="${toPx({ [xKey]: 0, [yKey]: 0 }).y.toFixed(2)}" r="${Math.max(goalRadiusPx, 3).toFixed(2)}" fill="none" stroke="#5cf084" stroke-width="1.5" stroke-dasharray="5 5"/>
  <path d="${path}" fill="none" stroke="#f5cd5c" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
  ${burnMarkers.map((point) => `<circle class="burn-marker" cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="3.2" fill="#5ce0a0"/>`).join("\n  ")}
  <circle cx="${start.x.toFixed(2)}" cy="${start.y.toFixed(2)}" r="5" fill="#60aee0"/>
  <circle cx="${end.x.toFixed(2)}" cy="${end.y.toFixed(2)}" r="5" fill="#f55c5c"/>
  <text x="${margin}" y="${height - 14}" fill="#9fb0c7" font-family="Inter, Arial, sans-serif" font-size="11">X: ${normalizedPlane === "RC" ? "C" : "I"} km | Y: R km | green dots: burn samples</text>
</svg>
`;
}

export function keplerianToEci(coes, muKm3S2 = DEFAULT_PURSUIT_CHALLENGE.mu_km3_s2) {
  const a = positiveNumber(coes.a_km, "a_km");
  const e = Number(coes.ecc || 0);
  const inc = degToRad(coes.inc_deg || 0);
  const raan = degToRad(coes.raan_deg || 0);
  const argp = degToRad(coes.argp_deg || 0);
  const nu = degToRad(coes.true_anomaly_deg || 0);
  const p = a * (1 - e * e);
  const radius = p / (1 + e * Math.cos(nu));
  const rPqw = [radius * Math.cos(nu), radius * Math.sin(nu), 0];
  const speedScale = Math.sqrt(muKm3S2 / p);
  const vPqw = [-speedScale * Math.sin(nu), speedScale * (e + Math.cos(nu)), 0];
  return {
    r: pqwToEci(rPqw, inc, raan, argp),
    v: pqwToEci(vPqw, inc, raan, argp),
  };
}

export function relativeRicState(target, chaser) {
  const basis = ricBasis(target);
  const relR = sub(chaser.r, target.r);
  const relV = sub(chaser.v, target.v);
  const omega = scale(basis.cHat, norm(cross(target.r, target.v)) / Math.max(norm(target.r) ** 2, DEFAULT_EPSILON));
  const rotatingRelV = sub(relV, cross(omega, relR));
  return {
    r_km: dot(relR, basis.rHat),
    i_km: dot(relR, basis.iHat),
    c_km: dot(relR, basis.cHat),
    rd_km_s: dot(rotatingRelV, basis.rHat),
    id_km_s: dot(rotatingRelV, basis.iHat),
    cd_km_s: dot(rotatingRelV, basis.cHat),
  };
}

export function stateFromRelativeRic(target, rel) {
  const basis = ricBasis(target);
  const relR = add(add(scale(basis.rHat, rel.r_km || 0), scale(basis.iHat, rel.i_km || 0)), scale(basis.cHat, rel.c_km || 0));
  const relVRot = add(add(scale(basis.rHat, rel.rd_km_s || 0), scale(basis.iHat, rel.id_km_s || 0)), scale(basis.cHat, rel.cd_km_s || 0));
  const omega = scale(basis.cHat, norm(cross(target.r, target.v)) / Math.max(norm(target.r) ** 2, DEFAULT_EPSILON));
  return {
    r: add(target.r, relR),
    v: add(target.v, add(relVRot, cross(omega, relR))),
  };
}

export function stepControlledTwoBodyPair({
  target,
  chaser,
  mu_km3_s2 = DEFAULT_PURSUIT_CHALLENGE.mu_km3_s2,
  chaser_accel_ric_km_s2 = [0, 0, 0],
  target_accel_ric_km_s2 = [0, 0, 0],
  dt_s = DEFAULT_PURSUIT_CHALLENGE.dt_s,
} = {}) {
  const pair = rk4Step(
    normalizeEciState(target),
    normalizeEciState(chaser),
    positiveNumber(mu_km3_s2, "mu_km3_s2"),
    vector3(chaser_accel_ric_km_s2, "chaser_accel_ric_km_s2"),
    vector3(target_accel_ric_km_s2, "target_accel_ric_km_s2"),
    positiveNumber(dt_s, "dt_s"),
  );
  return {
    target: normalizeEciState(pair.target),
    chaser: normalizeEciState(pair.chaser),
  };
}

export function stepTwoBodyState(
  state,
  mu_km3_s2 = DEFAULT_PURSUIT_CHALLENGE.mu_km3_s2,
  dt_s = DEFAULT_PURSUIT_CHALLENGE.dt_s,
) {
  return normalizeEciState(
    twoBodyStep(
      normalizeEciState(state),
      positiveNumber(mu_km3_s2, "mu_km3_s2"),
      positiveNumber(dt_s, "dt_s"),
    ),
  );
}

export function samplePursuitInitialRic(config = DEFAULT_PURSUIT_CHALLENGE, seed = 1) {
  const cfg = normalizeChallengeConfig(config);
  return cloneCanonical(sampleArcadeInitialRic(cfg, mulberry32(integerSeed(seed))));
}

export function deterministicSeed(seed, index = 0, salt = 0) {
  return roundSeed(integerSeed(seed), Math.max(Math.floor(Number(index) || 0), 0), Math.floor(Number(salt) || 0));
}

export function ellipticLinearCoastStates(rel0, timesS, chiefStateEci, muKm3S2 = DEFAULT_PURSUIT_CHALLENGE.mu_km3_s2) {
  const times = Array.isArray(timesS) ? timesS.map((time, idx) => ({ time: Math.max(Number(time) || 0, 0), idx })) : [];
  if (times.length === 0) return [];
  const chief = normalizeEciState(chiefStateEci);
  const rel = relativeArray(rel0);
  const sorted = times.slice().sort((a, b) => a.time - b.time);
  const results = new Array(times.length);
  let y = [...chief.r, ...chief.v, ...rel];
  let currentTime = 0;
  sorted.forEach(({ time, idx }) => {
    y = integrateEllipticLinearState(y, currentTime, time, muKm3S2);
    currentTime = time;
    results[idx] = {
      r: y[6],
      i: y[7],
      c: y[8],
      rd: y[9],
      id: y[10],
      cd: y[11],
      t: time,
    };
  });
  return results;
}

export function hashCanonicalJson(value) {
  const text = canonicalJson(value);
  let hash = HASH_OFFSET;
  for (let idx = 0; idx < text.length; idx += 1) {
    hash ^= text.charCodeAt(idx);
    hash = Math.imul(hash, HASH_PRIME) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

export function canonicalJson(value) {
  return JSON.stringify(cloneCanonical(value));
}

function stepArcadeState(sim, cfg, controls, dtS, prng) {
  const playerControl = clampControls(controls);
  const playerAccelRic = scale([playerControl.r, playerControl.i, playerControl.c], cfg.max_player_accel_km_s2);
  const targetAccelRic = targetDefenseAccelRic(sim, cfg, prng, dtS);
  const next = rk4Step(sim.target, sim.chaser, cfg.mu_km3_s2, playerAccelRic, targetAccelRic, dtS);
  const targetReference = twoBodyStep(sim.target_reference || sim.target, cfg.mu_km3_s2, dtS);
  return {
    ...sim,
    tick: sim.tick + 1,
    time_s: sim.time_s + dtS,
    target_reference: targetReference,
    target: next.target,
    chaser: next.chaser,
    player_delta_v_m_s: sim.player_delta_v_m_s + norm(playerAccelRic) * dtS * 1000,
    target_delta_v_m_s: sim.target_delta_v_m_s + norm(targetAccelRic) * dtS * 1000,
    target_pulse_remaining_s: Math.max(sim.target_pulse_remaining_s - dtS, 0),
  };
}

function rk4Step(target, chaser, mu, playerAccelRic, targetAccelRic, dtS) {
  const y0 = [...target.r, ...target.v, ...chaser.r, ...chaser.v];
  const derivative = (y) => {
    const targetState = { r: y.slice(0, 3), v: y.slice(3, 6) };
    const chaserState = { r: y.slice(6, 9), v: y.slice(9, 12) };
    const basis = ricBasis(targetState);
    const targetAccel = add(gravityAccel(targetState.r, mu), ricVectorToEci(targetAccelRic, basis));
    const chaserAccel = add(gravityAccel(chaserState.r, mu), ricVectorToEci(playerAccelRic, basis));
    return [...targetState.v, ...targetAccel, ...chaserState.v, ...chaserAccel];
  };
  const k1 = derivative(y0);
  const k2 = derivative(addScaled(y0, k1, dtS / 2));
  const k3 = derivative(addScaled(y0, k2, dtS / 2));
  const k4 = derivative(addScaled(y0, k3, dtS));
  const y = y0.map((value, idx) => value + (dtS / 6) * (k1[idx] + 2 * k2[idx] + 2 * k3[idx] + k4[idx]));
  return {
    target: { r: y.slice(0, 3), v: y.slice(3, 6) },
    chaser: { r: y.slice(6, 9), v: y.slice(9, 12) },
  };
}

function twoBodyStep(state, mu, dtS) {
  const y0 = [...state.r, ...state.v];
  const derivative = (y) => {
    const r = y.slice(0, 3);
    const v = y.slice(3, 6);
    return [...v, ...gravityAccel(r, mu)];
  };
  const k1 = derivative(y0);
  const k2 = derivative(addScaled(y0, k1, dtS / 2));
  const k3 = derivative(addScaled(y0, k2, dtS / 2));
  const k4 = derivative(addScaled(y0, k3, dtS));
  const y = y0.map((value, idx) => value + (dtS / 6) * (k1[idx] + 2 * k2[idx] + 2 * k3[idx] + k4[idx]));
  return { r: y.slice(0, 3), v: y.slice(3, 6) };
}

function integrateEllipticLinearState(y0, startTimeS, stopTimeS, mu) {
  let y = y0.slice();
  const durationS = Math.max(Number(stopTimeS) - Number(startTimeS), 0);
  if (durationS <= 0) return y;
  const steps = Math.max(1, Math.ceil(durationS / 60));
  const dtS = durationS / steps;
  for (let idx = 0; idx < steps; idx += 1) {
    const k1 = ellipticLinearDerivative(y, mu);
    const k2 = ellipticLinearDerivative(addScaled(y, k1, dtS / 2), mu);
    const k3 = ellipticLinearDerivative(addScaled(y, k2, dtS / 2), mu);
    const k4 = ellipticLinearDerivative(addScaled(y, k3, dtS), mu);
    y = y.map((value, stateIdx) => value + (dtS / 6) * (k1[stateIdx] + 2 * k2[stateIdx] + 2 * k3[stateIdx] + k4[stateIdx]));
  }
  return y;
}

function ellipticLinearDerivative(y, mu) {
  const chiefR = y.slice(0, 3);
  const chiefV = y.slice(3, 6);
  const rho = y.slice(6, 9);
  const rhoDot = y.slice(9, 12);
  const radius = Math.max(norm(chiefR), DEFAULT_EPSILON);
  const hVec = cross(chiefR, chiefV);
  const thetaDot = norm(hVec) / radius ** 2;
  const radialRate = dot(chiefR, chiefV) / radius;
  const thetaDdot = (-2 * thetaDot * radialRate) / radius;
  const omega = [0, 0, thetaDot];
  const omegaDot = [0, 0, thetaDdot];
  const gravityScale = mu / radius ** 3;
  const gravityGradient = [2 * gravityScale * rho[0], -gravityScale * rho[1], -gravityScale * rho[2]];
  const coriolis = scale(cross(omega, rhoDot), 2);
  const angularAccel = cross(omegaDot, rho);
  const centripetal = cross(omega, cross(omega, rho));
  const rhoDdot = sub(sub(sub(gravityGradient, coriolis), angularAccel), centripetal);
  return [...chiefV, ...gravityAccel(chiefR, mu), ...rhoDot, ...rhoDdot];
}

function targetDefenseAccelRic(sim, cfg, prng, dtS) {
  const defense = cfg.target_defense || {};
  if (!defense.enabled || sim.target_delta_v_m_s >= defense.max_delta_v_m_s) return [0, 0, 0];
  const rel = relativeRicState(sim.target, sim.chaser);
  const rangeKm = Math.hypot(rel.r_km, rel.i_km, rel.c_km);
  const closingKmS = -rangeRate(rel);
  const active = rangeKm < defense.trigger_range_km || closingKmS > defense.trigger_closing_speed_km_s;
  if (!active) return [0, 0, 0];
  const remainingBudget = Math.max(defense.max_delta_v_m_s - sim.target_delta_v_m_s, 0) / 1000;
  const accel = Math.min(defense.max_accel_km_s2, remainingBudget / Math.max(dtS, DEFAULT_EPSILON));
  return scale(unit(sim.target_pulse_direction_ric || seededFixedDirection(prng)), accel);
}

function initialArcadeState(cfg) {
  const target = keplerianToEci(cfg.target_coes, cfg.mu_km3_s2);
  const rel = cfg.chaser_initial_ric || {};
  const chaser = stateFromRelativeRic(target, rel);
  return {
    target_reference: {
      r: target.r.slice(),
      v: target.v.slice(),
    },
    target,
    chaser,
  };
}

function pursuitScore(cfg, result) {
  if (!result.passed) return 0;
  const difficultyMultiplier = difficultyScoreMultiplier(cfg.difficulty);
  const timeUsed = result.achieved_time_s === null || result.achieved_time_s === undefined ? cfg.max_time_s : result.achieved_time_s;
  const secondsRemaining = Math.max(cfg.max_time_s - timeUsed, 0);
  const chaserRemainingMmS = Math.max(cfg.max_delta_v_m_s - result.player_delta_v_m_s, 0) * 1000;
  const targetBudget = cfg.max_target_delta_v_m_s ?? cfg.target_defense?.max_delta_v_m_s;
  const targetRemainingMmS = targetBudget === null || targetBudget === undefined
    ? 0
    : Math.max(Number(targetBudget) - result.target_delta_v_m_s, 0) * 1000;
  return Math.max(0, Math.round((secondsRemaining + chaserRemainingMmS + targetRemainingMmS) * difficultyMultiplier));
}

function arcadeRoundConfig(baseConfig, { arcadeSeed, roundIndex, remainingTimeS }) {
  let cfg = cloneCanonical(baseConfig);
  cfg.max_time_s = Number(remainingTimeS);
  cfg.goal_range_km = arcadeRoundGoalRange(baseConfig, roundIndex);
  cfg.max_target_delta_v_m_s = arcadeTargetDeltaVBudget(baseConfig, roundIndex);
  cfg.target_defense = {
    ...(cfg.target_defense || {}),
    max_delta_v_m_s: arcadeTargetDeltaVBudget(baseConfig, roundIndex),
  };

  const initialRng = mulberry32(roundSeed(arcadeSeed, roundIndex, 715827883));
  if (arcadeRoundIsBoss(baseConfig, roundIndex)) {
    cfg.target_coes = arcadeBossTargetCoes(baseConfig, roundIndex, initialRng);
  } else {
    cfg.target_coes = cloneCanonical(baseConfig.target_coes);
  }
  if (roundIndex > 1 && cfg.arcade?.random_initial_state?.enabled) {
    cfg.chaser_initial_ric = sampleArcadeInitialRic(cfg, initialRng);
  } else {
    cfg.chaser_initial_ric = cloneCanonical(baseConfig.chaser_initial_ric);
  }
  return normalizeChallengeConfig(cfg);
}

function arcadeRoundGoalRange(baseConfig, roundIndex) {
  const base = Number(baseConfig.goal_range_km);
  const step = Number(baseConfig.arcade?.goal_range_step_km || 0);
  const minRange = Number(baseConfig.arcade?.min_goal_range_km || 0);
  return Math.max(base - Math.max(roundIndex - 1, 0) * step, minRange);
}

function arcadeRoundIsBoss(baseConfig, roundIndex) {
  const interval = Math.max(Math.floor(Number(baseConfig.arcade?.boss_round_interval || 0)), 0);
  return interval > 0 && roundIndex > 0 && roundIndex % interval === 0;
}

function arcadeRoundWeightedScore(roundConfig, result, roundIndex) {
  const multiplier = arcadeRoundIsBoss(roundConfig, roundIndex)
    ? Math.max(Number(roundConfig.arcade?.boss?.score_multiplier || 1), 0)
    : 1;
  return Math.round(Math.max(roundIndex, 1) * Number(result.score || 0) * multiplier);
}

function arcadeRoundTimeBonus(baseConfig, roundConfig, result, roundIndex) {
  const orbitalPeriodBonusS = 0.75 * targetOrbitalPeriodS(roundConfig);
  const baseline =
    Number(baseConfig.arcade?.round_bonus_time_s || 0) +
    orbitalPeriodBonusS +
    (arcadeRoundIsBoss(baseConfig, roundIndex) ? Number(baseConfig.arcade?.boss?.bonus_time_s || 0) : 0);
  const remainingDv = Math.max(Number(roundConfig.max_delta_v_m_s || 0) - Number(result.metrics?.player_delta_v_m_s || 0), 0);
  return baseline + remainingDv * Number(baseConfig.arcade?.delta_v_bonus_time_per_m_s || 0);
}

function targetOrbitalPeriodS(config) {
  const mu = positiveNumber(config?.mu_km3_s2 ?? DEFAULT_PURSUIT_CHALLENGE.mu_km3_s2, "mu_km3_s2");
  const aKm = positiveNumber(config?.target_coes?.a_km ?? DEFAULT_PURSUIT_CHALLENGE.target_coes.a_km, "target_coes.a_km");
  return 2 * Math.PI * Math.sqrt((aKm ** 3) / mu);
}

function arcadeTargetDeltaVBudget(baseConfig, roundIndex) {
  const defense = baseConfig.target_defense || {};
  const baseBudget = Number(defense.max_delta_v_m_s ?? baseConfig.max_target_delta_v_m_s ?? 0);
  const afterRound = Math.max(Math.floor(Number(defense.delta_v_ramp_after_round || 0)), 0);
  const step = Math.max(Number(defense.delta_v_ramp_step_m_s || 0), 0);
  return Math.max(baseBudget + Math.max(roundIndex - afterRound, 0) * step, 0);
}

function arcadeBossTargetCoes(baseConfig, roundIndex, rng) {
  const boss = baseConfig.arcade?.boss || {};
  const interval = Math.max(Math.floor(Number(baseConfig.arcade?.boss_round_interval || 0)), 1);
  const bossNumber = Math.max(Math.floor(roundIndex / interval), 1);
  const targetCoes = cloneCanonical(boss.target_coes || baseConfig.target_coes);
  const start = Number(boss.eccentricity_start ?? targetCoes.ecc ?? 0.05);
  const step = Number(boss.eccentricity_step || 0);
  const maxEcc = Number(boss.eccentricity_max ?? start);
  targetCoes.ecc = Math.min(start + Math.max(bossNumber - 1, 0) * step, maxEcc);
  const anomalyRange = rangePair(boss.true_anomaly_range_deg, [0, 360]);
  targetCoes.true_anomaly_deg = anomalyRange[0] + rng() * (anomalyRange[1] - anomalyRange[0]);
  return targetCoes;
}

function sampleArcadeInitialRic(cfg, rng) {
  const raw = cfg.arcade?.random_initial_state || {};
  const radialRange = rangePair(raw.radial_range_km, [-1, 1]);
  const inTrackRange = rangePair(raw.in_track_range_km, [-10, 10]);
  const crossTrackRange = rangePair(raw.cross_track_range_km, [-1, 1]);
  const crossTrackRateRange = rangePair(raw.cross_track_rate_range_km_s, [-0.001, 0.001]);
  const minRangeKm = Math.max(Number(raw.min_range_km || 5), 0);
  const target = keplerianToEci(cfg.target_coes, cfg.mu_km3_s2);
  for (let idx = 0; idx < 1000; idx += 1) {
    const relPosition = [
      uniformRange(rng, radialRange),
      uniformRange(rng, inTrackRange),
      uniformRange(rng, crossTrackRange),
    ];
    if (Math.hypot(...relPosition) < minRangeKm) continue;
    const cd = uniformRange(rng, crossTrackRateRange);
    const id = energyMatchedInTrackRate(relPosition, cd, target, cfg.mu_km3_s2);
    if (Number.isFinite(id)) {
      return {
        r_km: relPosition[0],
        i_km: relPosition[1],
        c_km: relPosition[2],
        rd_km_s: 0,
        id_km_s: id,
        cd_km_s: cd,
      };
    }
  }
  return cloneCanonical(cfg.chaser_initial_ric);
}

function energyMatchedInTrackRate(relPosition, crossTrackRateKmS, target, mu) {
  const baseRel = {
    r_km: relPosition[0],
    i_km: relPosition[1],
    c_km: relPosition[2],
    rd_km_s: 0,
    id_km_s: 0,
    cd_km_s: crossTrackRateKmS,
  };
  const unitRel = { ...baseRel, id_km_s: 1 };
  const baseState = stateFromRelativeRic(target, baseRel);
  const unitState = stateFromRelativeRic(target, unitRel);
  const vAxis = sub(unitState.v, baseState.v);
  const targetEnergy = specificOrbitalEnergy(target.r, target.v, mu);
  const qa = 0.5 * dot(vAxis, vAxis);
  const qb = dot(baseState.v, vAxis);
  const qc = 0.5 * dot(baseState.v, baseState.v) - mu / Math.max(norm(baseState.r), DEFAULT_EPSILON) - targetEnergy;
  if (qa <= 0) return NaN;
  const discriminant = qb * qb - 4 * qa * qc;
  if (discriminant < 0) return NaN;
  const root = Math.sqrt(Math.max(discriminant, 0));
  const candidates = [(-qb - root) / (2 * qa), (-qb + root) / (2 * qa)];
  return Math.abs(candidates[0]) <= Math.abs(candidates[1]) ? candidates[0] : candidates[1];
}

function specificOrbitalEnergy(r, v, mu) {
  return 0.5 * dot(v, v) - mu / Math.max(norm(r), DEFAULT_EPSILON);
}

function roundSeed(seed, roundIndex, salt) {
  return hashNumbersToSeed([integerSeed(seed), Math.max(Math.floor(roundIndex), 1), Math.floor(salt)]);
}

function hashNumbersToSeed(values) {
  let hash = HASH_OFFSET;
  values.forEach((value) => {
    hash ^= Number(value) >>> 0;
    hash = Math.imul(hash, HASH_PRIME) >>> 0;
  });
  return hash >>> 0;
}

function rangePair(value, fallback) {
  if (!Array.isArray(value) || value.length < 2) return fallback.slice();
  const lower = Number(value[0]);
  const upper = Number(value[1]);
  if (!Number.isFinite(lower) || !Number.isFinite(upper)) return fallback.slice();
  return lower <= upper ? [lower, upper] : [upper, lower];
}

function uniformRange(rng, pair) {
  return pair[0] + rng() * (pair[1] - pair[0]);
}

function difficultyScoreMultiplier(difficulty) {
  const key = String(difficulty || "easy").trim().toLowerCase();
  if (key === "medium" || key === "normal") return 2;
  if (key === "hard") return 3;
  if (key === "extreme" || key === "expert") return 4;
  return 1;
}

function compareClaimedResult(attempt, replay, options) {
  const errors = [];
  const scoreTolerance = Number(options.score_tolerance ?? 0);
  if (Math.abs(Number(attempt.claimed_score || 0) - replay.score) > scoreTolerance) {
    errors.push(`claimed_score ${attempt.claimed_score} does not match canonical score ${replay.score}.`);
  }
  const claimed = attempt.claimed_metrics || {};
  const tolerances = {
    elapsed_s: Number(options.elapsed_tolerance_s ?? 0.5),
    player_delta_v_m_s: Number(options.delta_v_tolerance_m_s ?? 1.0e-6),
    closest_range_km: Number(options.range_tolerance_km ?? 1.0e-6),
    final_range_km: Number(options.range_tolerance_km ?? 1.0e-6),
  };
  Object.entries(tolerances).forEach(([key, tolerance]) => {
    if (claimed[key] === undefined || replay.metrics[key] === null) return;
    if (Math.abs(Number(claimed[key]) - Number(replay.metrics[key])) > tolerance) {
      errors.push(`claimed metric ${key}=${claimed[key]} does not match canonical ${replay.metrics[key]}.`);
    }
  });
  return errors;
}

function compareClaimedArcadeResult(attempt, replay, options) {
  const errors = [];
  const scoreTolerance = Number(options.score_tolerance ?? 0);
  if (Math.abs(Number(attempt.claimed_score || 0) - replay.score) > scoreTolerance) {
    errors.push(`claimed_score ${attempt.claimed_score} does not match canonical score ${replay.score}.`);
  }
  const claimed = attempt.claimed_metrics || {};
  const tolerances = {
    rounds_cleared: 0,
    final_round_index: 0,
    remaining_time_s: Number(options.elapsed_tolerance_s ?? 0.5),
    player_delta_v_m_s: Number(options.delta_v_tolerance_m_s ?? 1.0e-6),
    target_delta_v_m_s: Number(options.delta_v_tolerance_m_s ?? 1.0e-6),
    final_range_km: Number(options.range_tolerance_km ?? 1.0e-6),
    final_relative_speed_km_s: Number(options.speed_tolerance_km_s ?? 1.0e-9),
  };
  Object.entries(tolerances).forEach(([key, tolerance]) => {
    if (claimed[key] === undefined || replay.metrics[key] === null || replay.metrics[key] === undefined) return;
    if (Math.abs(Number(claimed[key]) - Number(replay.metrics[key])) > tolerance) {
      errors.push(`claimed metric ${key}=${claimed[key]} does not match canonical ${replay.metrics[key]}.`);
    }
  });
  return errors;
}

function arcadeRoundAttemptFromSession(session, roundIndex) {
  const result = session.result();
  const snap = session.snapshot();
  return {
    round_index: Math.max(1, Math.floor(Number(roundIndex || 1))),
    final_tick: Math.max(0, Math.floor(Number(result.ticks ?? snap.tick ?? 0))),
    input_events: normalizeInputEvents(snap.input_events || []),
  };
}

function arcadeRoundAttemptsWithCurrent(roundAttempts, session, roundIndex) {
  const attempts = normalizeRoundAttempts(roundAttempts || []);
  const snap = session.snapshot();
  const last = attempts[attempts.length - 1];
  if (snap.passed && last?.round_index === Math.max(1, Math.floor(Number(roundIndex || 1))) - 1) return attempts;
  const current = arcadeRoundAttemptFromSession(session, roundIndex);
  if (!last || last.round_index !== current.round_index) attempts.push(current);
  return normalizeRoundAttempts(attempts);
}

function arcadeRunMetrics(snap, attempts) {
  const finalRoundIndex = snap.round_transition?.cleared_round_index ?? snap.round_index;
  return {
    rounds_cleared: Math.max(0, (snap.round_summaries || []).length),
    final_round_index: Math.max(1, Math.floor(Number(finalRoundIndex || 1))),
    remaining_time_s: roundMetric(snap.remaining_time_s),
    final_range_km: roundMetric(snap.range_km),
    final_relative_speed_km_s: roundMetric(snap.relative_speed_km_s),
    player_delta_v_m_s: roundMetric(snap.player_delta_v_m_s),
    target_delta_v_m_s: roundMetric(snap.target_delta_v_m_s),
    round_attempts: attempts.length,
  };
}

function normalizeRoundAttempts(rounds) {
  if (!Array.isArray(rounds)) throw new Error("round_attempts must be an array.");
  return rounds.map((round, idx) => ({
    round_index: Math.max(1, Math.floor(Number(round.round_index ?? round.roundIndex ?? idx + 1))),
    final_tick: Math.max(0, Math.floor(Number(round.final_tick ?? round.finalTick ?? 0))),
    input_events: normalizeInputEvents(round.input_events || round.inputEvents || []),
  }));
}

function validateRoundAttempts(rounds, baseConfig) {
  if (!Array.isArray(rounds)) throw new Error("round_attempts must be an array.");
  if (rounds.length < 1) throw new Error("round_attempts must include at least one round.");
  if (rounds.length > 200) throw new Error("round_attempts exceeds the maximum supported arcade run length.");
  const looseTimeConfig = {
    ...baseConfig,
    max_time_s: Math.max(Number(baseConfig.max_time_s || 0), Number(baseConfig.arcade?.initial_time_s || 0), 7 * 86400),
  };
  rounds.forEach((round, idx) => {
    const expectedRoundIndex = idx + 1;
    if (round.round_index !== expectedRoundIndex) {
      throw new Error(`round_attempts must be contiguous from round 1; found round ${round.round_index}.`);
    }
    if (!Number.isInteger(round.final_tick) || round.final_tick < 0) {
      throw new Error(`Invalid final_tick for round ${round.round_index}: ${round.final_tick}.`);
    }
    const maxTick = Math.ceil(looseTimeConfig.max_time_s / looseTimeConfig.dt_s) + 1;
    if (round.final_tick > maxTick) {
      throw new Error(`final_tick for round ${round.round_index} exceeds the arcade time budget.`);
    }
    validateInputEvents(round.input_events, looseTimeConfig);
  });
}

function normalizeInputEvents(events) {
  if (!Array.isArray(events)) throw new Error("input_events must be an array.");
  return events.map((event) => ({
    tick: Math.floor(Number(event.tick)),
    control: String(event.control || ""),
    state: String(event.state || "").toLowerCase(),
  }));
}

function validateInputEvents(events, cfg) {
  let lastTick = -1;
  const maxTick = Math.ceil(cfg.max_time_s / cfg.dt_s) + 1;
  events.forEach((event) => {
    if (!Number.isInteger(event.tick) || event.tick < 0 || event.tick > maxTick) {
      throw new Error(`Invalid input event tick: ${event.tick}.`);
    }
    if (event.tick < lastTick) throw new Error("input_events must be sorted by tick.");
    lastTick = event.tick;
    if (!CONTROL_IDS.includes(event.control)) throw new Error(`Unknown control id: ${event.control}.`);
    if (!["down", "up"].includes(event.state)) throw new Error(`Invalid control state: ${event.state}.`);
  });
}

function eventsByTick(events) {
  const map = new Map();
  events.forEach((event) => {
    if (!map.has(event.tick)) map.set(event.tick, []);
    map.get(event.tick).push(event);
  });
  return map;
}

function applyTickEvents(activeControls, events) {
  const downThisTick = new Set();
  const pulseControls = new Set();
  events.forEach((event) => {
    if (event.state === "down") {
      activeControls.add(event.control);
      downThisTick.add(event.control);
    }
    if (event.state === "up") {
      if (downThisTick.has(event.control)) pulseControls.add(event.control);
      activeControls.delete(event.control);
    }
  });
  return pulseControls;
}

function controlsFromActive(activeControls, pulseControls = new Set()) {
  const isActive = (control) => activeControls.has(control) || pulseControls.has(control);
  const r = Number(isActive("rPlus")) - Number(isActive("rMinus"));
  const i = Number(isActive("iPlus")) - Number(isActive("iMinus"));
  const c = Number(isActive("cPlus")) - Number(isActive("cMinus"));
  return clampControls({ r, i, c });
}

function clampControls(controls) {
  const r = Number(controls.r || 0);
  const i = Number(controls.i || 0);
  const c = Number(controls.c || 0);
  const mag = Math.hypot(r, i, c);
  if (mag <= 1) return { r, i, c };
  return { r: r / mag, i: i / mag, c: c / mag };
}

function vector3(value, name) {
  if (!Array.isArray(value) || value.length !== 3) throw new Error(`${name} must contain exactly three values.`);
  const result = value.map(Number);
  if (!result.every(Number.isFinite)) throw new Error(`${name} must contain finite values.`);
  return result;
}

function historySample(sim, rel, activeControls, pulseControls = new Set()) {
  const reference = sim.target_reference || sim.target;
  const targetReferenceRel = relativeRicState(reference, sim.target);
  const chaserReferenceRel = relativeRicState(reference, sim.chaser);
  return {
    tick: sim.tick,
    time_s: roundMetric(sim.time_s),
    relative_ric: {
      r_km: roundMetric(rel.r_km),
      i_km: roundMetric(rel.i_km),
      c_km: roundMetric(rel.c_km),
      rd_km_s: roundMetric(rel.rd_km_s),
      id_km_s: roundMetric(rel.id_km_s),
      cd_km_s: roundMetric(rel.cd_km_s),
    },
    target_reference_ric: {
      r_km: roundMetric(targetReferenceRel.r_km),
      i_km: roundMetric(targetReferenceRel.i_km),
      c_km: roundMetric(targetReferenceRel.c_km),
      rd_km_s: roundMetric(targetReferenceRel.rd_km_s),
      id_km_s: roundMetric(targetReferenceRel.id_km_s),
      cd_km_s: roundMetric(targetReferenceRel.cd_km_s),
    },
    chaser_reference_ric: {
      r_km: roundMetric(chaserReferenceRel.r_km),
      i_km: roundMetric(chaserReferenceRel.i_km),
      c_km: roundMetric(chaserReferenceRel.c_km),
      rd_km_s: roundMetric(chaserReferenceRel.rd_km_s),
      id_km_s: roundMetric(chaserReferenceRel.id_km_s),
      cd_km_s: roundMetric(chaserReferenceRel.cd_km_s),
    },
    target_reference_state_eci: eciStateBlock(reference),
    controls: controlsFromActive(activeControls, pulseControls),
    player_delta_v_m_s: roundMetric(sim.player_delta_v_m_s),
    target_delta_v_m_s: roundMetric(sim.target_delta_v_m_s),
  };
}

function eciStateBlock(state) {
  return {
    r_km: state.r.map(roundMetric),
    v_km_s: state.v.map(roundMetric),
  };
}

function normalizeEciState(state) {
  const r = state?.r || state?.r_km || state?.position_eci_km;
  const v = state?.v || state?.v_km_s || state?.velocity_eci_km_s;
  if (!Array.isArray(r) || r.length !== 3 || !Array.isArray(v) || v.length !== 3) {
    throw new Error("chiefStateEci must include 3-element r/v vectors.");
  }
  return {
    r: r.map(Number),
    v: v.map(Number),
  };
}

function relativeArray(rel) {
  if (Array.isArray(rel)) return rel.slice(0, 6).map(Number);
  return [
    Number(rel?.r_km ?? rel?.r ?? 0),
    Number(rel?.i_km ?? rel?.i ?? 0),
    Number(rel?.c_km ?? rel?.c ?? 0),
    Number(rel?.rd_km_s ?? rel?.rd ?? 0),
    Number(rel?.id_km_s ?? rel?.id ?? 0),
    Number(rel?.cd_km_s ?? rel?.cd ?? 0),
  ];
}

function compactBurnMarkers(markers) {
  const compact = [];
  let lastTick = -Infinity;
  markers.forEach((marker) => {
    if (marker.tick - lastTick < 3) return;
    compact.push({
      tick: marker.tick,
      time_s: roundMetric(marker.time_s),
      controls: marker.controls,
      relative_ric: {
        r_km: roundMetric(marker.relative_ric.r_km),
        i_km: roundMetric(marker.relative_ric.i_km),
        c_km: roundMetric(marker.relative_ric.c_km),
      },
    });
    lastTick = marker.tick;
  });
  return compact;
}

function seededFixedDirection(prng) {
  return unit([randomNormal(prng), randomNormal(prng), randomNormal(prng)]);
}

function randomNormal(prng) {
  const u1 = Math.max(prng(), 1.0e-12);
  const u2 = prng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function rangeRate(rel) {
  const range = Math.hypot(rel.r_km, rel.i_km, rel.c_km);
  if (range <= DEFAULT_EPSILON) return 0;
  return (rel.r_km * rel.rd_km_s + rel.i_km * rel.id_km_s + rel.c_km * rel.cd_km_s) / range;
}

function ricBasis(state) {
  const rHat = unit(state.r);
  const cHat = unit(cross(state.r, state.v));
  const iHat = unit(cross(cHat, rHat));
  return { rHat, iHat, cHat };
}

function ricVectorToEci([r, i, c], basis) {
  return add(add(scale(basis.rHat, r), scale(basis.iHat, i)), scale(basis.cHat, c));
}

function gravityAccel(r, mu) {
  const radius = Math.max(norm(r), DEFAULT_EPSILON);
  return scale(r, -mu / radius ** 3);
}

function pqwToEci(vec, inc, raan, argp) {
  const cosO = Math.cos(raan);
  const sinO = Math.sin(raan);
  const cosI = Math.cos(inc);
  const sinI = Math.sin(inc);
  const cosW = Math.cos(argp);
  const sinW = Math.sin(argp);
  const m11 = cosO * cosW - sinO * sinW * cosI;
  const m12 = -cosO * sinW - sinO * cosW * cosI;
  const m21 = sinO * cosW + cosO * sinW * cosI;
  const m22 = -sinO * sinW + cosO * cosW * cosI;
  const m31 = sinW * sinI;
  const m32 = cosW * sinI;
  return [
    m11 * vec[0] + m12 * vec[1],
    m21 * vec[0] + m22 * vec[1],
    m31 * vec[0] + m32 * vec[1],
  ];
}

function plotExtents(points, xKey, yKey, minSpanKm) {
  const xs = points.length ? points.map((point) => point[xKey]) : [0];
  const ys = points.length ? points.map((point) => point[yKey]) : [0];
  let minX = Math.min(...xs, 0);
  let maxX = Math.max(...xs, 0);
  let minY = Math.min(...ys, 0);
  let maxY = Math.max(...ys, 0);
  const span = Math.max(maxX - minX, maxY - minY, minSpanKm);
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  const padded = span * 1.15;
  minX = midX - padded / 2;
  maxX = midX + padded / 2;
  minY = midY - padded / 2;
  maxY = midY + padded / 2;
  return { minX, maxX, minY, maxY };
}

function cloneCanonical(value) {
  if (Array.isArray(value)) return value.map(cloneCanonical);
  if (value && typeof value === "object") {
    return Object.keys(value)
      .sort()
      .reduce((acc, key) => {
        acc[key] = cloneCanonical(value[key]);
        return acc;
      }, {});
  }
  return value;
}

function deepFreeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    Object.values(value).forEach(deepFreeze);
  }
  return value;
}

function mulberry32(seed) {
  let state = integerSeed(seed);
  return function random() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function integerSeed(seed) {
  const value = Number(seed);
  if (!Number.isFinite(value)) return 1;
  return Math.floor(value) >>> 0;
}

function roundMetric(value) {
  if (!Number.isFinite(value)) return null;
  return Number(value.toPrecision(12));
}

function rangeFromRelativeSample(sample) {
  const rel = sample?.relative_ric;
  if (!rel) return Infinity;
  return Math.hypot(rel.r_km, rel.i_km, rel.c_km);
}

function formatSeconds(seconds) {
  return `${Number(seconds || 0).toFixed(1)} s`;
}

function formatMps(value) {
  return `${Number(value || 0).toFixed(3)} m/s`;
}

function escapeXml(value) {
  return String(value).replace(/[<>&"']/g, (char) => ({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "\"": "&quot;",
    "'": "&apos;",
  })[char]);
}

function degToRad(value) {
  return (Number(value) * Math.PI) / 180;
}

function add(a, b) {
  return a.map((value, idx) => value + b[idx]);
}

function sub(a, b) {
  return a.map((value, idx) => value - b[idx]);
}

function scale(a, scalar) {
  return a.map((value) => value * scalar);
}

function dot(a, b) {
  return a.reduce((total, value, idx) => total + value * b[idx], 0);
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(a) {
  return Math.hypot(...a);
}

function unit(a) {
  const mag = Math.max(norm(a), DEFAULT_EPSILON);
  return scale(a, 1 / mag);
}

function addScaled(a, b, scalar) {
  return a.map((value, idx) => value + b[idx] * scalar);
}

function positiveNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new Error(`${name} must be positive.`);
  return number;
}

function isFiniteNumber(value) {
  return Number.isFinite(Number(value));
}
