import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  advanceFrameTiming,
  createFrameTimingState,
  MAX_PENDING_SIMULATION_S,
  reconcileFrameTimingExecution,
  resetFrameTimingState,
} from "../src/frame-timing.js";

const appSource = readFileSync(
  fileURLToPath(new URL("../src/app.js", import.meta.url)),
  "utf8",
);

function approximately(actual, expected, tolerance = 1.0e-9) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

test("60 Hz 200x coast keeps fractional pending work and bounds frame steps", () => {
  const timing = createFrameTimingState();
  const first = advanceFrameTiming(timing, {
    elapsedS: 1 / 60,
    speedMultiple: 200,
    stepDtS: 0.1,
    maxStepsPerFrame: 32,
  });

  assert.equal(first.steps, 32);
  approximately(first.executedSimulationS, 3.2);
  approximately(first.pendingSimulationS, 200 / 60 - 3.2);
  assert.equal(first.droppedSimulationS, 0);

  for (let frame = 1; frame < 120; frame += 1) {
    const plan = advanceFrameTiming(timing, {
      elapsedS: 1 / 60,
      speedMultiple: 200,
      stepDtS: 0.1,
      maxStepsPerFrame: 32,
    });
    assert.ok(plan.steps <= 32);
    assert.ok(plan.pendingSimulationS >= 0);
    assert.ok(plan.pendingSimulationS <= MAX_PENDING_SIMULATION_S);
  }

  assert.ok(timing.droppedSimulationS > 0);
  assert.ok(timing.actualRateMultiple > 0);
});

test("a 200 ms frame reports dropped work after the bounded 32-step budget", () => {
  const timing = createFrameTimingState();
  const plan = advanceFrameTiming(timing, {
    elapsedS: 0.2,
    speedMultiple: 200,
    stepDtS: 0.1,
    maxStepsPerFrame: 32,
  });

  assert.equal(plan.steps, 32);
  approximately(plan.requestedSimulationS, 40);
  approximately(plan.admittedSimulationS, 40);
  approximately(plan.clippedSimulationS, 0);
  approximately(plan.droppedSimulationS, 32);
  approximately(plan.pendingSimulationS, 4.8);
  approximately(plan.currentActualRateMultiple, 16);
  approximately(timing.actualRateMultiple, 16);
  approximately(plan.requestedSimulationS, plan.executedSimulationS + plan.pendingSimulationS + plan.droppedSimulationS);
});

test("a long suspension reports all clipped and capped simulation work", () => {
  const timing = createFrameTimingState();
  const plan = advanceFrameTiming(timing, {
    elapsedS: 12,
    wallElapsedS: 12,
    speedMultiple: 200,
    stepDtS: 0.1,
    maxStepsPerFrame: 32,
  });

  assert.equal(plan.steps, 32);
  approximately(plan.requestedSimulationS, 2400);
  approximately(plan.admittedSimulationS, 40);
  approximately(plan.clippedSimulationS, 2360);
  approximately(plan.pendingDropSimulationS, 32);
  approximately(plan.droppedSimulationS, 2392);
  approximately(plan.pendingSimulationS, 4.8);
  assert.ok(plan.pendingSimulationS <= MAX_PENDING_SIMULATION_S);
  approximately(timing.droppedSimulationS, 2392);
  approximately(plan.currentActualRateMultiple, 3.2 / 12);
  approximately(timing.actualRateMultiple, 3.2 / 12);
  approximately(plan.requestedSimulationS, plan.executedSimulationS + plan.pendingSimulationS + plan.droppedSimulationS);
});

test("terminal completion refunds planned no-op steps and preserves conservation", () => {
  const timing = createFrameTimingState();
  const plan = advanceFrameTiming(timing, {
    elapsedS: 0.2,
    speedMultiple: 200,
    stepDtS: 0.1,
    maxStepsPerFrame: 32,
  });
  const execution = reconcileFrameTimingExecution(timing, {
    plannedSteps: plan.steps,
    actualSteps: 1,
    stepDtS: 0.1,
  });

  assert.equal(execution.actualSteps, 1);
  approximately(execution.refundedSimulationS, 3.1);
  approximately(execution.pendingSimulationS, 7.9);
  approximately(timing.executedSimulationS, 0.1);
  approximately(timing.executedSimulationThisFrameS, 0.1);
  approximately(timing.currentActualRateMultiple, 0.5);
  approximately(timing.actualRateMultiple, 0.5);
  approximately(timing.requestedSimulationS, timing.executedSimulationS + timing.pendingSimulationS + timing.droppedSimulationS);
});

test("paused frames preserve pending fractional work without inventing drops", () => {
  const timing = createFrameTimingState();
  const running = advanceFrameTiming(timing, {
    elapsedS: 1 / 60,
    speedMultiple: 1,
    stepDtS: 0.1,
    maxStepsPerFrame: 32,
  });
  const paused = advanceFrameTiming(timing, {
    elapsedS: 2,
    speedMultiple: 200,
    stepDtS: 0.1,
    maxStepsPerFrame: 32,
    shouldRun: false,
  });

  approximately(paused.pendingSimulationS, running.pendingSimulationS);
  assert.equal(paused.steps, 0);
  assert.equal(paused.droppedSimulationS, 0);
  assert.equal(timing.droppedSimulationS, 0);
});

test("reset clears pending, drop, and achieved-rate counters", () => {
  const timing = createFrameTimingState();
  advanceFrameTiming(timing, {
    elapsedS: 0.2,
    speedMultiple: 200,
    stepDtS: 0.1,
    maxStepsPerFrame: 32,
  });
  resetFrameTimingState(timing);

  assert.deepEqual(timing, createFrameTimingState());
});

test("app retains the 10x maneuver rail while using bounded frame timing", () => {
  assert.match(appSource, /const MANEUVER_CONTROL_SPEED = 10;/);
  assert.match(appSource, /return MANEUVER_CONTROL_SPEED;/);
  assert.match(appSource, /maxStepsPerFrame: MAX_STEPS_PER_FRAME/);
  assert.match(appSource, /maxPendingSimulationS: MAX_PENDING_SIMULATION_S/);
  assert.match(appSource, /timingResetToken/);
  assert.match(appSource, /reconcileFrameTimingExecution/);
  assert.match(appSource, /state\.passed \|\| state\.timingResetToken !== timingResetToken/);
  assert.doesNotMatch(appSource, /steps >= MAX_STEPS_PER_FRAME\)\s*\{\s*state\.stepAccumulatorS = 0/s);
});
