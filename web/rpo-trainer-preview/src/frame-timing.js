/**
 * Deterministic browser-frame scheduling for fixed-step preview simulation.
 *
 * The renderer may execute only a bounded number of physics steps per browser
 * frame. The remainder stays pending so the fractional step is preserved. A
 * finite frame admission interval and pending bound protect the UI after a
 * long tab suspension; all clipped or capped work is explicitly reported.
 */

export const FRAME_ELAPSED_CAP_S = 0.2;
export const MAX_PENDING_SIMULATION_S = 8.0;

const EPSILON_S = 1.0e-9;

function finiteNonNegative(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? numeric : fallback;
}

function positiveFinite(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
}

export function createFrameTimingState() {
  return {
    pendingSimulationS: 0,
    droppedSimulationS: 0,
    droppedSimulationThisFrameS: 0,
    clippedSimulationThisFrameS: 0,
    admittedSimulationS: 0,
    requestedSimulationS: 0,
    executedSimulationS: 0,
    executedSimulationThisFrameS: 0,
    wallElapsedS: 0,
    wallElapsedThisFrameS: 0,
    actualRateMultiple: 0,
    currentActualRateMultiple: 0,
  };
}

export function resetFrameTimingState(timing) {
  const target = timing && typeof timing === "object" ? timing : createFrameTimingState();
  Object.assign(target, createFrameTimingState());
  return target;
}

/**
 * Consume one browser-frame budget and return the number of fixed steps to run.
 *
 * ``pendingSimulationS`` may be supplied when a caller keeps a compatibility
 * alias for the accumulator. ``wallElapsedS`` is the uncapped interval used
 * to account for the full request, while ``elapsedS`` is the bounded interval
 * admitted to this frame. The returned ``pendingSimulationS`` always includes
 * the fractional remainder after the bounded step budget.
 */
export function advanceFrameTiming(
  timing,
  {
    pendingSimulationS,
    elapsedS = 0,
    wallElapsedS = elapsedS,
    speedMultiple = 1,
    stepDtS,
    maxStepsPerFrame = 32,
    maxPendingSimulationS = MAX_PENDING_SIMULATION_S,
    shouldRun = true,
  } = {},
) {
  const target = timing && typeof timing === "object" ? timing : createFrameTimingState();
  const step = positiveFinite(stepDtS, 1);
  const maxSteps = Math.max(Math.floor(finiteNonNegative(maxStepsPerFrame, 0)), 0);
  const maxPending = positiveFinite(maxPendingSimulationS, MAX_PENDING_SIMULATION_S);
  const boundedElapsed = Math.min(finiteNonNegative(elapsedS), FRAME_ELAPSED_CAP_S);
  const wallElapsed = finiteNonNegative(wallElapsedS);
  const speed = finiteNonNegative(speedMultiple, 0);
  const previousPending = finiteNonNegative(pendingSimulationS ?? target.pendingSimulationS);
  const requested = shouldRun ? wallElapsed * speed : 0;
  const admitted = shouldRun ? Math.min(requested, boundedElapsed * speed) : 0;
  const clipped = Math.max(requested - admitted, 0);
  let pending = previousPending + admitted;
  const pendingDrop = Math.max(pending - maxPending, 0);
  if (pendingDrop > 0) pending = maxPending;
  const dropped = clipped + pendingDrop;

  const steps = shouldRun
    ? Math.min(Math.floor((pending + EPSILON_S) / step), maxSteps)
    : 0;
  const executed = steps * step;
  pending = Math.max(pending - executed, 0);
  const currentRate = shouldRun && wallElapsed > EPSILON_S ? executed / wallElapsed : 0;

  target.pendingSimulationS = pending;
  target.droppedSimulationThisFrameS = dropped;
  target.clippedSimulationThisFrameS = clipped;
  target.admittedSimulationS = finiteNonNegative(target.admittedSimulationS) + admitted;
  target.droppedSimulationS = finiteNonNegative(target.droppedSimulationS) + dropped;
  target.requestedSimulationS = finiteNonNegative(target.requestedSimulationS) + requested;
  target.executedSimulationS = finiteNonNegative(target.executedSimulationS) + executed;
  target.executedSimulationThisFrameS = executed;
  target.wallElapsedS = finiteNonNegative(target.wallElapsedS) + (shouldRun ? wallElapsed : 0);
  target.wallElapsedThisFrameS = shouldRun ? wallElapsed : 0;
  target.currentActualRateMultiple = currentRate;
  target.actualRateMultiple =
    target.wallElapsedS > EPSILON_S ? target.executedSimulationS / target.wallElapsedS : 0;

  return {
    steps,
    pendingSimulationS: pending,
    requestedSimulationS: requested,
    admittedSimulationS: admitted,
    executedSimulationS: executed,
    wallElapsedS: shouldRun ? wallElapsed : 0,
    clippedSimulationS: clipped,
    pendingDropSimulationS: pendingDrop,
    droppedSimulationS: dropped,
    currentActualRateMultiple: currentRate,
    actualRateMultiple: target.actualRateMultiple,
  };
}

/**
 * Refund fixed-step work that was planned but could not be advanced because a
 * terminal state was reached during the frame. This keeps timing evidence
 * conservative and preserves the pending fraction for an explicit lifecycle
 * reset or inspection instead of counting no-op steps as executed simulation.
 */
export function reconcileFrameTimingExecution(timing, { plannedSteps = 0, actualSteps = 0, stepDtS } = {}) {
  const target = timing && typeof timing === "object" ? timing : createFrameTimingState();
  const step = positiveFinite(stepDtS, 1);
  const planned = Math.max(Math.floor(finiteNonNegative(plannedSteps, 0)), 0);
  const actual = Math.min(Math.max(Math.floor(finiteNonNegative(actualSteps, 0)), 0), planned);
  const refunded = (planned - actual) * step;
  if (refunded <= 0) {
    return { actualSteps: actual, refundedSimulationS: 0, pendingSimulationS: target.pendingSimulationS };
  }

  target.pendingSimulationS = finiteNonNegative(target.pendingSimulationS) + refunded;
  target.executedSimulationS = Math.max(finiteNonNegative(target.executedSimulationS) - refunded, 0);
  target.executedSimulationThisFrameS = Math.max(
    finiteNonNegative(target.executedSimulationThisFrameS) - refunded,
    0,
  );
  target.currentActualRateMultiple =
    target.wallElapsedThisFrameS > EPSILON_S
      ? target.executedSimulationThisFrameS / target.wallElapsedThisFrameS
      : 0;
  target.actualRateMultiple =
    target.wallElapsedS > EPSILON_S ? target.executedSimulationS / target.wallElapsedS : 0;

  return { actualSteps: actual, refundedSimulationS: refunded, pendingSimulationS: target.pendingSimulationS };
}
